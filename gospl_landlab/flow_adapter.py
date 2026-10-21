"""Map goSPL flow fields onto Landlab, or run a Landlab flow router."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .loader import resolve_landlab_class


def ensure_flow_fields(grid) -> None:
    n = grid.number_of_nodes
    defaults = {
        "flow__receiver_node": np.arange(n, dtype=int),
        "flow__upstream_node_order": np.arange(n, dtype=int),
        "flow__link_to_receiver_node": np.full(n, -1, dtype=int),
        "topographic__steepest_slope": np.zeros(n, dtype=float),
        "surface_water__discharge": np.zeros(n, dtype=float),
        "drainage_area": np.zeros(n, dtype=float),
        "water__unit_flux_in": np.ones(n, dtype=float),
    }
    for name, val in defaults.items():
        if name not in grid.at_node:
            grid.add_field(name, val.copy(), at="node", clobber=True)


def map_gospl_flow_to_landlab(
    mirror,
    *,
    discharge: np.ndarray,
    receiver_steepest: np.ndarray,
    slope: Optional[np.ndarray] = None,
    rainfall: float = 1.0,
) -> None:
    """
    Inject goSPL discharge + single-receiver graph into Landlab node fields.

    Parameters
    ----------
    discharge :
        Volumetric water discharge at goSPL nodes (m³/yr), gathered global order.
    receiver_steepest :
        Local/global goSPL receiver index per node (SFD / steepest of MFD).
        Values are indices into the **same gathered array**.
    """
    grid = mirror.grid
    ensure_flow_fields(grid)
    g2l = mirror.gospl_to_ll

    q = np.asarray(discharge, dtype=float).ravel()
    rcv = np.asarray(receiver_steepest, dtype=int).ravel()
    if q.size != g2l.size or rcv.size != g2l.size:
        raise ValueError("discharge/receiver size must match gathered goSPL nodes")

    # Map receivers into Landlab node IDs
    rcv_ll = np.arange(grid.number_of_nodes, dtype=int)
    valid = (rcv >= 0) & (rcv < g2l.size)
    rcv_ll[g2l[valid]] = g2l[rcv[valid]]
    # self-receivers stay self
    self_mask = ~valid
    rcv_ll[g2l[self_mask]] = g2l[self_mask]

    grid.at_node["flow__receiver_node"][:] = rcv_ll
    grid.at_node["surface_water__discharge"][g2l] = q
    # drainage area proxy: Q / P (P in m/yr)
    p = max(float(rainfall), 1e-12)
    grid.at_node["drainage_area"][g2l] = q / p
    grid.at_node["water__unit_flux_in"][:] = p

    if slope is not None:
        s = np.asarray(slope, dtype=float).ravel()
        grid.at_node["topographic__steepest_slope"][g2l] = s
    else:
        # crude fallback from elev + receiver
        z = grid.at_node["topographic__elevation"]
        dz = z - z[rcv_ll]
        # avoid div0; Landlab dx if raster
        dx = float(getattr(mirror, "dx", None) or 1.0)
        grid.at_node["topographic__steepest_slope"][:] = np.maximum(dz, 0.0) / dx

    # upstream order: topological sort of single-receiver DAG (donors before receivers → reverse)
    grid.at_node["flow__upstream_node_order"][:] = _upstream_order(rcv_ll)

    # link to receiver: best-effort on raster
    try:
        links = grid.links_at_node
        dirs = grid.link_dirs_at_node
        link_to_rcv = np.full(grid.number_of_nodes, -1, dtype=int)
        for i in range(grid.number_of_nodes):
            r = rcv_ll[i]
            if r == i:
                continue
            for j, link in enumerate(links[i]):
                if link < 0:
                    continue
                # head/tail depending on direction
                if dirs[i, j] == -1 and grid.node_at_link_head[link] == r:
                    link_to_rcv[i] = link
                    break
                if dirs[i, j] == 1 and grid.node_at_link_tail[link] == r:
                    link_to_rcv[i] = link
                    break
        grid.at_node["flow__link_to_receiver_node"][:] = link_to_rcv
    except Exception:
        pass


def _upstream_order(receiver: np.ndarray) -> np.ndarray:
    """Return downstream-to-upstream order (Landlab convention for many erosers)."""
    n = receiver.size
    indeg = np.zeros(n, dtype=int)
    for i, r in enumerate(receiver):
        if r != i and 0 <= r < n:
            indeg[r] += 1
    # Kahn from heads (indeg 0) toward outlets — then reverse for upstream order
    from collections import deque

    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    seen = np.zeros(n, dtype=bool)
    while q:
        i = q.popleft()
        if seen[i]:
            continue
        seen[i] = True
        order.append(i)
        r = receiver[i]
        if r != i and 0 <= r < n:
            indeg[r] -= 1
            if indeg[r] == 0:
                q.append(r)
    for i in range(n):
        if not seen[i]:
            order.append(i)
    # Landlab wants downstream-to-upstream: reverse
    return np.asarray(order[::-1], dtype=int)


def _set_node_field(grid, name: str, arr: np.ndarray) -> None:
    """Add or replace a node field (handles shape change)."""
    if name in grid.at_node:
        existing = grid.at_node[name]
        if getattr(existing, "shape", None) == arr.shape:
            existing[:] = arr
            return
        grid.delete_field("node", name)
    grid.add_field(name, arr, at="node", clobber=True)


# Per live-grid caches: avoid rebuilding scratch MFD / DepressionFinder every step.
_mfd_scratch_cache: Dict[int, Dict[str, Any]] = {}
_depression_finder_cache: Dict[int, Dict[str, Any]] = {}


def clear_hylands_routing_caches() -> None:
    """Drop cached scratch MFD grids and DepressionFinder instances (tests / remesh)."""
    _mfd_scratch_cache.clear()
    _depression_finder_cache.clear()


def _mfd_scratch_entry(grid) -> Dict[str, Any]:
    """Return cached scratch RasterModelGrid + FlowDirectorMFD for ``grid``."""
    from landlab import RasterModelGrid
    from landlab.components import FlowDirectorMFD

    key = id(grid)
    shape = tuple(grid.shape)
    spacing = tuple(np.atleast_1d(grid.spacing).astype(float).tolist())
    entry = _mfd_scratch_cache.get(key)
    if (
        entry is None
        or entry.get("live") is not grid
        or entry.get("shape") != shape
        or entry.get("spacing") != spacing
    ):
        scratch = RasterModelGrid(grid.shape, xy_spacing=grid.spacing)
        if "topographic__elevation" not in scratch.at_node:
            scratch.add_zeros("topographic__elevation", at="node")
        mfd = FlowDirectorMFD(scratch, diagonals=True)
        entry = {
            "live": grid,
            "shape": shape,
            "spacing": spacing,
            "scratch": scratch,
            "mfd": mfd,
        }
        _mfd_scratch_cache[key] = entry
    return entry


def _depression_finder_entry(grid) -> Dict[str, Any]:
    """Return cached DepressionFinderAndRouter bound to ``grid``."""
    from landlab.components import DepressionFinderAndRouter

    key = id(grid)
    entry = _depression_finder_cache.get(key)
    if entry is None or entry.get("live") is not grid:
        entry = {
            "live": grid,
            "df": DepressionFinderAndRouter(grid),
        }
        _depression_finder_cache[key] = entry
    return entry


def update_hill_flow_from_mfd(grid) -> None:
    """
    Build HyLands ``hill_flow__*`` from Landlab ``FlowDirectorMFD`` (8-neighbor).

    Runs MFD on a **scratch** raster with only elevation copied — never deletes or
    recreates SFD ``flow__*`` / steepest-slope on ``grid``. SPACE (and FA) keep
    cached views into those arrays; delete/restore used to orphan them and kill
    fluvial carving.

    Scratch grid + MFD director are cached per live grid (shape/spacing); each call
    only copies status/elev and re-runs ``run_one_step``.
    """
    from landlab import RasterModelGrid

    if not isinstance(grid, RasterModelGrid):
        raise TypeError(
            "update_hill_flow_from_mfd expects a RasterModelGrid "
            f"(got {type(grid).__name__})"
        )

    entry = _mfd_scratch_entry(grid)
    scratch = entry["scratch"]
    # Match open/closed perimeter so MFD edges behave like the live grid
    scratch.status_at_node[:] = grid.status_at_node
    scratch.at_node["topographic__elevation"][:] = grid.at_node[
        "topographic__elevation"
    ]

    entry["mfd"].run_one_step()
    hill_rcv = np.array(scratch.at_node["flow__receiver_node"], copy=True)
    hill_prop = np.array(scratch.at_node["flow__receiver_proportions"], copy=True)
    hill_slope = np.array(scratch.at_node["topographic__steepest_slope"], copy=True)

    _set_node_field(grid, "hill_flow__receiver_node", hill_rcv)
    _set_node_field(grid, "hill_flow__receiver_proportions", hill_prop)
    _set_node_field(grid, "hill_topographic__steepest_slope", hill_slope)


def update_flood_status(grid) -> None:
    """
    Set ``flood_status_code`` via Landlab ``DepressionFinderAndRouter``.

    goSPL has no Landlab lake/flood mask; do not invent all-zeros. Requires
    single-receiver ``flow__receiver_node`` already on the grid (goSPL map or FA).

    DepressionFinder instance is cached per live grid; ``map_depressions`` still
    runs each call (elev changes).
    """
    if "flow__receiver_node" not in grid.at_node:
        raise RuntimeError(
            "update_flood_status needs flow__receiver_node "
            "(map goSPL FA or run Landlab flow first)"
        )
    rcv = np.asarray(grid.at_node["flow__receiver_node"])
    if getattr(rcv, "ndim", 1) != 1:
        raise RuntimeError(
            "update_flood_status expects 1D SFD flow__receiver_node; "
            f"got shape {getattr(rcv, 'shape', None)}"
        )

    # DepressionFinder reads flow__sink_flag (FlowAccumulator normally writes it)
    nodes = np.arange(rcv.size, dtype=int)
    sinks = rcv == nodes
    if "flow__sink_flag" not in grid.at_node:
        grid.add_field("flow__sink_flag", sinks.copy(), at="node", clobber=True)
    else:
        grid.at_node["flow__sink_flag"][:] = sinks

    df = _depression_finder_entry(grid)["df"]
    df.map_depressions()
    _set_node_field(
        grid, "flood_status_code", np.asarray(df.flood_status, dtype=int)
    )


def prepare_hylands_routing(grid) -> None:
    """MFD hill flow + depression flood mask for ``BedrockLandslider``."""
    update_hill_flow_from_mfd(grid)
    update_flood_status(grid)


def run_landlab_flow(mirror, component_name: str, kwargs: Optional[Dict[str, Any]] = None):
    """Instantiate and run a Landlab flow router on the mirror grid."""
    cls = resolve_landlab_class(component_name)
    kwargs = dict(kwargs or {})
    fr = cls(mirror.grid, **kwargs)
    # PriorityFloodFlowRouter / FlowAccumulator
    if hasattr(fr, "run_one_step"):
        try:
            fr.run_one_step()
        except TypeError:
            fr.run_one_step(1.0)
    elif hasattr(fr, "update"):
        fr.update()
    return fr
