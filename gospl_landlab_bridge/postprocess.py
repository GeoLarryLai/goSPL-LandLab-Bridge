"""Postprocess helpers: elev/erodep maps, mid-transect strata, progress panels, GIFs.

Strata fills use paleo-elevation / water-depth bins (foreland-style), not grain size.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Depositional elev thresholds (m, high→low) + fills for strata cross-sections
_PALEO_DEPTH_ID = (50.0, 20.0, 0.0, -20.0, -50.0)
_PALEO_COLORS = (
    "#F4D03F",  # >50 yellow
    "limegreen",  # 20–50 green
    "sandybrown",  # 0–20 (warm / “pink”)
    "darkkhaki",  # 0 to −20 (previous shallow-negative fill)
    "#2E86C1",  # −20 to −50 medium blue
    "#1A5276",  # ≤−50 darker blue
)
_PALEO_LABELS = (
    ">50 m",
    "20–50 m",
    "0–20 m",
    "−20–0 m",
    "−50––20 m",
    "≤−50 m",
)
_DEPO_CBAR_LIM = (-120.0, 120.0)  # linear colorbar axis (m)
# Fallback when paleo elev missing: cycle soft pastels by layer index
_STRATA_COLORS = _PALEO_COLORS + ("wheat", "darkseagreen")
_LINE_INTERVAL_YR = 0.5e6
_SEA_LEVEL_COLOR = "lightblue"
_SEA_LEVEL_ALPHA = 0.3
_TITLE_A = "A: goSPL SPL + LD"
_TITLE_B = "B: goSPL SPL; Landlab NLD + HyLands"
_CBAR_ELEV = "Elevation (m)"
_CBAR_CUMED = "Cumulative erosion/deposition (m)"
# Hillshade elev→raster: thread chunks + mesh reuse across GIF frames (A/B).
_HILLSHADE_N_JOBS = 10
_HILLSHADE_CACHE_MAX = 2
_HILLSHADE_MESH_CACHE: Dict[tuple, dict] = {}


def latest_run_dir(base: Path) -> Path:
    """Prefer ``base`` if it has outputs; else newest ``base_N`` with HDF5 steps."""

    def score(p: Path) -> int:
        return len(list((p / "h5").glob("gospl.*.p0.h5"))) if (p / "h5").exists() else 0

    base = Path(base)
    cands = [base] + sorted(base.parent.glob(base.name + "_*"), key=lambda p: p.stat().st_mtime)
    cands = [p for p in cands if score(p) > 0]
    if not cands:
        raise FileNotFoundError(base)
    cands.sort(key=lambda p: (score(p), p.stat().st_mtime))
    return cands[-1]


def list_gospl_steps(run_dir: Path) -> List[int]:
    h5dir = Path(run_dir) / "h5"
    steps = []
    for p in h5dir.glob("gospl.*.p0.h5"):
        m = re.search(r"gospl\.(\d+)\.p0\.h5$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def resolve_gospl_h5(
    run_dir: Path,
    step: int,
    *,
    repair: bool = True,
) -> Optional[Path]:
    """
    Path to ``gospl.{step}.p0.h5``, or a cloud-sync conflicted copy if missing.

    When ``repair`` and only a conflicted copy exists, copy it back to the
    canonical name so later loads see a normal file.
    """
    import shutil

    h5dir = Path(run_dir) / "h5"
    if not h5dir.is_dir():
        return None
    canon = h5dir / f"gospl.{int(step)}.p0.h5"
    if canon.is_file() and canon.stat().st_size > 0:
        return canon

    # e.g. "gospl.8.p0 (Lai Larry S.'s conflicted copy 2026-08-09).h5"
    cands = [
        p
        for p in h5dir.glob(f"gospl.{int(step)}.p0*.h5")
        if p.is_file()
        and p.stat().st_size > 0
        and re.match(rf"gospl\.{int(step)}\.p0", p.name)
        and p.name != canon.name
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best = cands[0]
    if repair:
        try:
            shutil.copy2(best, canon)
            return canon
        except OSError:
            return best
    return best


def shared_existing_steps(
    run_a: Path,
    run_b: Path,
    steps: Optional[Sequence[int]] = None,
) -> List[int]:
    """Steps with readable goSPL HDF5 on both runs (repairs cloud-sync conflicts)."""
    run_a, run_b = Path(run_a), Path(run_b)
    if steps is None:
        steps = list_gospl_steps(run_a)
    out: List[int] = []
    for s in steps:
        pa = resolve_gospl_h5(run_a, int(s), repair=True)
        pb = resolve_gospl_h5(run_b, int(s), repair=True)
        if pa is not None and pb is not None:
            out.append(int(s))
    return out


def _read_text_retry(
    path: Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    retries: int = 6,
    base_sleep_s: float = 0.4,
) -> str:
    """``Path.read_text`` with retries for a cloud-sync ``TimeoutError`` (errno 60)."""
    import time

    path = Path(path)
    last: Optional[BaseException] = None
    for i in range(max(1, int(retries))):
        try:
            return path.read_text(encoding=encoding, errors=errors)
        except OSError as exc:
            last = exc
            # Cloud-synced folder: ETIMEDOUT while hydrating cloud-only files
            if getattr(exc, "errno", None) not in (60, "60") and not isinstance(
                exc, TimeoutError
            ):
                raise
            time.sleep(base_sleep_s * (i + 1))
    assert last is not None
    raise last


def _step_time_yr(
    run_dir: Path,
    step: int,
    *,
    dt_yr: Optional[float] = None,
) -> float:
    """Best-effort time for a step: XMF when readable, else ``step * dt_yr``."""
    t = read_step_time_yr(run_dir, step)
    if t is not None:
        return float(t)
    if dt_yr is not None and float(dt_yr) > 0:
        return float(step) * float(dt_yr)
    return float(step)


def read_step_time_yr(run_dir: Path, step: int) -> Optional[float]:
    """Parse ``Time Value`` from ``xmf/gospl{step}.xmf`` when present.

    Retries cloud-sync timeouts; returns ``None`` if the file is missing or still
    unreadable after retries (callers should fall back to ``step * dt``).
    """
    xmf = Path(run_dir) / "xmf" / f"gospl{step}.xmf"
    if not xmf.exists():
        return None
    try:
        text = _read_text_retry(xmf)
    except OSError:
        return None
    m = re.search(r'Time[^>]*Value="([0-9.eE+-]+)"', text)
    return float(m.group(1)) if m else None


def _orient_nodes_layers(arr: np.ndarray, n_nodes: int) -> np.ndarray:
    """Ensure ``(n_nodes, n_layers)``; transpose if goSPL stored the other way."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"expected 2D stratal array, got shape {a.shape}")
    if a.shape[0] == n_nodes:
        return a
    if a.shape[1] == n_nodes:
        return a.T
    raise ValueError(f"stratal shape {a.shape} does not match n_nodes={n_nodes}")


def sediment_interfaces_from_thickness(
    elev: np.ndarray,
    strat_h: np.ndarray,
    *,
    h_min: float = 1e-3,
    h_max_sane: float = 1.0e5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack positive ``stratH`` under the current surface (foreland-style packages).

    Returns
    -------
    interfaces : (n_layers+1, n_nodes)
        Bottom (basement) then tops after each layer; last ≈ ``elev``.
    thick : (n_layers, n_nodes)
        Cleaned thicknesses (zeros where missing / sentinel).
    """
    elev = np.asarray(elev, dtype=float).ravel()
    H = np.asarray(strat_h, dtype=float)
    if H.ndim != 2 or H.shape[0] != elev.size:
        raise ValueError(f"strat_h shape {H.shape} vs elev {elev.shape}")
    H = np.where(np.isfinite(H), H, 0.0)
    H = np.clip(H, 0.0, None)
    H[H > h_max_sane] = 0.0  # goSPL sometimes writes huge sentinels
    H[H < h_min] = 0.0

    n_nodes, n_lay = H.shape
    total = H.sum(axis=1)
    basement = elev - total
    interfaces = np.empty((n_lay + 1, n_nodes), dtype=float)
    interfaces[0] = basement
    for k in range(n_lay):
        interfaces[k + 1] = interfaces[k] + H[:, k]
    # pin top to elev (float drift / eroded bookkeeping)
    interfaces[-1] = elev
    return interfaces, H.T  # thick as (n_layers, n_nodes) for fill loops


def load_gospl_step(run_dir: Path, step: int) -> dict:
    """
    Load coordinates + elev/erodep (+ optional stratal) for one output step.

    Returns dict: x, y, cells (n_tri, 3) or None, elev, erodep,
    strata / strat_h (n_nodes, n_layers) or None, time_yr, path.
    """
    import h5py

    run_dir = Path(run_dir)
    h5dir = run_dir / "h5"
    path = resolve_gospl_h5(run_dir, int(step), repair=True)
    if path is None:
        raise FileNotFoundError(h5dir / f"gospl.{int(step)}.p0.h5")
    topo = h5dir / "topology.p0.h5"
    with h5py.File(topo, "r") as f:
        coords = np.asarray(f["coords"])
        cells = None
        if "cells" in f:
            cells = np.asarray(f["cells"], dtype=int)
            # goSPL stores 1-based triangle indices
            if cells.size and int(cells.min()) >= 1:
                cells = cells - 1
    with h5py.File(path, "r") as f:
        elev = np.asarray(f["elev"]).ravel()
        erodep = np.asarray(f["erodep"]).ravel()

    strata = None
    strat_h = None
    sfile = h5dir / f"stratal.{step}.p0.h5"
    if sfile.exists():
        with h5py.File(sfile, "r") as f:
            if "stratZ" in f:
                strata = _orient_nodes_layers(np.asarray(f["stratZ"]), elev.size)
            if "stratH" in f:
                strat_h = _orient_nodes_layers(np.asarray(f["stratH"]), elev.size)

    return {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cells": cells,
        "elev": elev,
        "erodep": erodep,
        "strata": strata,
        "strat_h": strat_h,
        "time_yr": read_step_time_yr(run_dir, step),
        "step": step,
        "path": path,
    }


def _triangulation(x_km: np.ndarray, y_km: np.ndarray, cells: Optional[np.ndarray]):
    """Build matplotlib Triangulation from goSPL cells (or Delaunay fallback)."""
    import matplotlib.tri as mtri

    x_km = np.asarray(x_km, dtype=float).ravel()
    y_km = np.asarray(y_km, dtype=float).ravel()
    if cells is not None:
        tri = np.asarray(cells, dtype=int)
        if tri.ndim == 2 and tri.shape[1] == 3:
            return mtri.Triangulation(x_km, y_km, tri)
    return mtri.Triangulation(x_km, y_km)


def _map_field(
    ax,
    x,
    y,
    v,
    *,
    cells=None,
    cmap="terrain",
    vmin=None,
    vmax=None,
):
    """Filled triangular map (no scatter dots / white gaps between nodes)."""
    x_km = np.asarray(x, dtype=float).ravel() / 1000.0
    y_km = np.asarray(y, dtype=float).ravel() / 1000.0
    v = np.asarray(v, dtype=float).ravel()
    tri = _triangulation(x_km, y_km, cells)
    return ax.tripcolor(
        tri,
        v,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def _hillshade_mesh_key(
    x_km: np.ndarray, y_km: np.ndarray, cells: Optional[np.ndarray]
) -> tuple:
    """Fingerprint fixed mesh so GIF frames can reuse tri/KDTree (A/B cache)."""
    n = int(x_km.size)
    mid = n // 2
    cfp: Optional[tuple]
    if cells is None:
        cfp = None
    else:
        c = np.asarray(cells, dtype=int)
        cfp = (
            c.shape,
            int(c.flat[0]),
            int(c.flat[-1]),
            int(c.flat[c.size // 2]),
        )
    return (
        n,
        float(x_km[0]),
        float(x_km[mid]),
        float(x_km[-1]),
        float(y_km[0]),
        float(y_km[mid]),
        float(y_km[-1]),
        round(float(np.nanmin(x_km)), 6),
        round(float(np.nanmax(x_km)), 6),
        round(float(np.nanmin(y_km)), 6),
        round(float(np.nanmax(y_km)), 6),
        cfp,
    )


def _get_hillshade_mesh(
    x_km: np.ndarray, y_km: np.ndarray, cells: Optional[np.ndarray]
) -> dict:
    """Cache triangulation + KDTree + raster grid (max 2 meshes: run A / run B)."""
    import matplotlib.tri as mtri
    from scipy.spatial import cKDTree

    key = _hillshade_mesh_key(x_km, y_km, cells)
    hit = _HILLSHADE_MESH_CACHE.get(key)
    if hit is not None:
        return hit

    pts = np.column_stack([x_km, y_km])
    tri = None
    if cells is not None:
        tri_idx = np.asarray(cells, dtype=int)
        if tri_idx.ndim == 2 and tri_idx.shape[1] == 3:
            tri = mtri.Triangulation(x_km, y_km, tri_idx)

    tree = cKDTree(pts)
    xmin, xmax = float(np.nanmin(x_km)), float(np.nanmax(x_km))
    ymin, ymax = float(np.nanmin(y_km)), float(np.nanmax(y_km))
    n = pts.shape[0]
    sample = pts[:: max(1, n // 2000)]
    d, _ = tree.query(sample, k=2)
    dx_km = float(np.median(d[:, 1])) * 0.5
    if not np.isfinite(dx_km) or dx_km <= 0:
        dx_km = max((xmax - xmin), (ymax - ymin)) / 200.0
    dx_km = max(dx_km, 0.05)

    nx = max(2, int(np.ceil((xmax - xmin) / dx_km)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / dx_km)) + 1)
    while nx * ny > 2_000_000:
        dx_km *= 1.5
        nx = max(2, int(np.ceil((xmax - xmin) / dx_km)) + 1)
        ny = max(2, int(np.ceil((ymax - ymin) / dx_km)) + 1)

    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    XI, YI = np.meshgrid(xi, yi)
    entry = {
        "pts": pts,
        "tri": tri,
        "tree": tree,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "dx_km": dx_km,
        "XI": XI,
        "YI": YI,
    }
    while len(_HILLSHADE_MESH_CACHE) >= _HILLSHADE_CACHE_MAX:
        oldest = next(iter(_HILLSHADE_MESH_CACHE))
        del _HILLSHADE_MESH_CACHE[oldest]
    _HILLSHADE_MESH_CACHE[key] = entry
    return entry


def _hillshade_eval_chunk(interp, XI_chunk: np.ndarray, YI_chunk: np.ndarray) -> np.ndarray:
    """Evaluate interpolator on one row-block; masked → NaN."""
    out = np.asarray(interp(XI_chunk, YI_chunk), dtype=float)
    if np.ma.isMaskedArray(out):
        out = np.ma.filled(out, np.nan)
    return out


def _hillshade_eval_parallel(interp, XI: np.ndarray, YI: np.ndarray) -> np.ndarray:
    """Split raster rows into ``_HILLSHADE_N_JOBS`` chunks; thread-parallel eval."""
    from joblib import Parallel, delayed

    ny = int(XI.shape[0])
    n_jobs = max(1, min(int(_HILLSHADE_N_JOBS), ny))
    bounds = np.linspace(0, ny, n_jobs + 1, dtype=int)
    chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_hillshade_eval_chunk)(interp, XI[a:b], YI[a:b])
        for a, b in zip(bounds[:-1], bounds[1:])
        if b > a
    )
    return np.vstack(chunks)


def _map_elev_hillshade(
    ax,
    x,
    y,
    v,
    *,
    cells=None,
    cmap="terrain",
    vmin=None,
    vmax=None,
):
    """
    Elevation map with hillshade under terrain colors (outside goSPL).

    Rasterizes node elev via ``LinearTriInterpolator`` (goSPL ``cells``) or
    ``LinearNDInterpolator`` fallback, fills exterior NaNs with one ``cKDTree``
    nearest query, shades with ``LightSource``, draws ``imshow``. Mesh
    triangulation / tree / grid are cached across GIF frames (max 2 meshes).
    Returns an unshaded ``ScalarMappable`` for the elevation colorbar.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import LightSource, Normalize
    from scipy.interpolate import LinearNDInterpolator

    x_km = np.asarray(x, dtype=float).ravel() / 1000.0
    y_km = np.asarray(y, dtype=float).ravel() / 1000.0
    v = np.asarray(v, dtype=float).ravel()
    mesh = _get_hillshade_mesh(x_km, y_km, cells)
    XI, YI = mesh["XI"], mesh["YI"]
    xmin, xmax = mesh["xmin"], mesh["xmax"]
    ymin, ymax = mesh["ymin"], mesh["ymax"]
    dx_km = mesh["dx_km"]

    if mesh["tri"] is not None:
        interp = mtri.LinearTriInterpolator(mesh["tri"], v)
    else:
        interp = LinearNDInterpolator(mesh["pts"], v)

    Z = _hillshade_eval_parallel(interp, XI, YI)
    miss = ~np.isfinite(Z)
    if np.any(miss):
        _, idx = mesh["tree"].query(
            np.column_stack([XI[miss], YI[miss]]),
            k=1,
        )
        Z[miss] = v[np.asarray(idx, dtype=int)]

    if vmin is None:
        vmin = float(np.nanmin(v))
    if vmax is None:
        vmax = float(np.nanmax(v))
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    dx_m = float(dx_km) * 1000.0
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        Z,
        cmap=cmap_obj,
        blend_mode="overlay",
        vmin=vmin,
        vmax=vmax,
        dx=dx_m,
        dy=dx_m,
        vert_exag=1.0,
    )
    ax.imshow(
        rgb,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        interpolation="bilinear",
        zorder=1,
    )
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_obj)
    sm.set_array([])
    return sm


def _map_elev(
    ax,
    x,
    y,
    v,
    *,
    cells=None,
    cmap="terrain",
    vmin=None,
    vmax=None,
):
    """Topography panel: hillshaded elevation colors."""
    return _map_elev_hillshade(
        ax, x, y, v, cells=cells, cmap=cmap, vmin=vmin, vmax=vmax
    )


def _overlay_mesh(
    ax,
    x,
    y,
    cells=None,
    *,
    color: str = "0.95",
    lw: float = 0.25,
    ms: float = 0.8,
):
    """Thin triangle edges + small node dots on top of a tripcolor map."""
    x_km = np.asarray(x, dtype=float).ravel() / 1000.0
    y_km = np.asarray(y, dtype=float).ravel() / 1000.0
    tri = _triangulation(x_km, y_km, cells)
    ax.triplot(tri, color=color, lw=lw, zorder=5)
    ax.plot(x_km, y_km, ".", color=color, markersize=ms, zorder=6)


def _fill_layer_by_paleo(
    ax,
    x_km: np.ndarray,
    y_bot: np.ndarray,
    y_top: np.ndarray,
    thick_row: np.ndarray,
    paleo_row: Optional[np.ndarray],
    layer_index: int,
) -> None:
    """Fill one stratum; color by paleo elev bins when available."""
    base_mask = np.asarray(thick_row, dtype=float) > 0.0
    if not np.any(base_mask):
        return
    if paleo_row is None:
        c = _STRATA_COLORS[layer_index % len(_STRATA_COLORS)]
        ax.fill_between(
            x_km,
            y_bot,
            y_top,
            where=base_mask,
            facecolor=c,
            linewidth=0,
            alpha=0.9,
            interpolate=True,
            zorder=1,
        )
        return
    paleo = np.asarray(paleo_row, dtype=float).ravel()
    edges = (np.inf,) + tuple(_PALEO_DEPTH_ID) + (-np.inf,)
    for j, c in enumerate(_PALEO_COLORS):
        hi, lo = edges[j], edges[j + 1]
        mask = base_mask & (paleo <= hi) & (paleo > lo)
        if not np.any(mask):
            continue
        ax.fill_between(
            x_km,
            y_bot,
            y_top,
            where=mask,
            facecolor=c,
            linewidth=0,
            alpha=0.9,
            interpolate=True,
            zorder=1,
        )


def _milestone_layer_indices(
    n_lay: int,
    *,
    layer_dt_yr: Optional[float],
    line_interval_yr: float = _LINE_INTERVAL_YR,
) -> set:
    """Layer tops (0-based) that fall on the black-line cadence (e.g. 0.5 Myr)."""
    if layer_dt_yr is None or layer_dt_yr <= 0 or n_lay <= 0:
        return set()
    n_per = max(1, int(round(float(line_interval_yr) / float(layer_dt_yr))))
    return {i for i in range(n_lay) if (i + 1) % n_per == 0}


def pick_progress_steps(steps: Sequence[int], n: int = 3) -> List[int]:
    """
    Pick ``n`` steps spanning the run (early / mid / late).

    Prefers skipping the initial dump when enough outputs exist.
    """
    steps = list(steps)
    if not steps:
        raise ValueError("no goSPL output steps")
    if len(steps) <= n:
        return steps
    # skip t≈0 dump when possible
    usable = steps[1:] if len(steps) > n else steps
    if len(usable) < n:
        usable = steps
    idx = np.linspace(0, len(usable) - 1, n)
    picked = [usable[int(round(i))] for i in idx]
    # unique, preserve order
    out: List[int] = []
    for s in picked:
        if s not in out:
            out.append(s)
    while len(out) < n and len(out) < len(steps):
        for s in steps:
            if s not in out:
                out.append(s)
            if len(out) >= n:
                break
    return out[:n]


def pick_steps_nearest_times(
    run_dir: Path,
    steps: Sequence[int],
    times_yr: Sequence[float],
    *,
    dt_yr: Optional[float] = None,
) -> List[int]:
    """Map target times (yr) to nearest available output step indices.

    Prefer XMF times; if ``dt_yr`` is set, use ``step * dt_yr`` and skip mass
    XMF reads (important on a cloud-synced folder). Otherwise fall back per-step with retry.
    """
    steps = list(steps)
    if not steps:
        raise ValueError("no goSPL output steps")
    if dt_yr is not None and float(dt_yr) > 0:
        timed = [(s, float(s) * float(dt_yr)) for s in steps]
    else:
        timed = [
            (s, _step_time_yr(run_dir, s, dt_yr=dt_yr)) for s in steps
        ]
    picked: List[int] = []
    for target in times_yr:
        s_best, _ = min(timed, key=lambda st: abs(st[1] - float(target)))
        if s_best not in picked:
            picked.append(s_best)
    return picked


def pick_steps_every_interval(
    run_dir: Path,
    steps: Sequence[int],
    interval_yr: float,
    *,
    dt_yr: Optional[float] = None,
) -> List[int]:
    """
    Subsample output steps nearest to 0, interval, 2*interval, … through the run end.

    Used so GIFs can show e.g. every 20 kyr while the model ran at 5 kyr.
    Pass ``dt_yr`` (model output stride) to avoid reading every XMF on a cloud-synced folder.
    """
    steps = list(steps)
    if not steps:
        raise ValueError("no goSPL output steps")
    interval_yr = float(interval_yr)
    if interval_yr <= 0:
        return steps
    if dt_yr is not None and float(dt_yr) > 0:
        timed = [(s, float(s) * float(dt_yr)) for s in steps]
    else:
        timed = [
            (s, _step_time_yr(run_dir, s, dt_yr=dt_yr)) for s in steps
        ]
    t_end = max(t for _, t in timed)
    targets = list(np.arange(0.0, t_end + 0.5 * interval_yr, interval_yr))
    if targets[-1] < t_end - 1e-6:
        targets.append(t_end)
    return pick_steps_nearest_times(
        run_dir, steps, targets, dt_yr=dt_yr
    )


def mid_transect_indices(x: np.ndarray, y: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    """Node indices along a mid-y east–west band (works for jittered meshes)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ym = 0.5 * (y.min() + y.max())
    if tol is None:
        from scipy.spatial import cKDTree

        pts = np.column_stack([x, y])
        n = pts.shape[0]
        step = max(1, n // 2000)
        sample = pts[::step]
        tree = cKDTree(pts)
        d, _ = tree.query(sample, k=2)
        spacing = float(np.median(d[:, 1]))
        tol = 0.6 * spacing if np.isfinite(spacing) and spacing > 0 else 1.0
    ids = np.where(np.abs(y - ym) <= tol)[0]
    if ids.size < 2:
        # widen once if the band is empty (very sparse mid-row)
        ids = np.where(np.abs(y - ym) <= 2.0 * tol)[0]
    order = np.argsort(x[ids])
    return ids[order]


def _time_label(time_yr: Optional[float], step: int) -> str:
    if time_yr is None:
        return f"t = step {step}"
    if time_yr >= 1e6:
        return f"t = {time_yr / 1e6:.2f} Myr"
    return f"t = {time_yr / 1e3:.0f} kyr"


def _panel_title(run: str, kind: str, time_yr: Optional[float], step: int) -> str:
    """Two-line panel title: A/B backend line, then field + time."""
    first = _TITLE_A if run.upper().startswith("A") else _TITLE_B
    tlab = _time_label(time_yr, step)
    if kind == "elev":
        second = f"Topography ({tlab})"
    else:
        second = f"Cumulative Erosion / Deposition ({tlab})"
    return f"{first}\n{second}"


def _add_cbar(fig, mappable, ax, label: str):
    """Colorbar matching the host axes height (right side)."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if isinstance(ax, (list, tuple, np.ndarray)):
        host = list(np.ravel(ax))[-1]
    else:
        host = ax
    cax = inset_axes(
        host,
        width="4.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=host.transAxes,
        borderpad=0,
    )
    return fig.colorbar(mappable, cax=cax, label=label)


def _add_paleo_legend(ax) -> None:
    """Paleo-depth color key (xs font); place once, typically top-left panel."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=c, edgecolor="none", label=lab)
        for c, lab in zip(_PALEO_COLORS, _PALEO_LABELS)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize="x-small",
        framealpha=0.85,
        title="Paleo depth",
        title_fontsize="x-small",
    )


def _depositional_cbar_bounds() -> List[float]:
    """Increasing BoundaryNorm edges: colorbar lims + bin thresholds."""
    lo, hi = float(_DEPO_CBAR_LIM[0]), float(_DEPO_CBAR_LIM[1])
    # _PALEO_DEPTH_ID is high→low; reverse for increasing bounds interior
    interior = sorted(float(t) for t in _PALEO_DEPTH_ID)
    return [lo] + interior + [hi]


def _depositional_scalar_mappable():
    """Shared stepped colormap for depositional elev (−120…+120 m axis)."""
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm, ListedColormap

    bounds = _depositional_cbar_bounds()
    colors = list(reversed(_PALEO_COLORS))  # low→high for increasing bounds
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=len(colors), clip=True)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm, bounds


def _add_paleo_cbar(fig, ax, label: str = "Depositional elev / depth (m)"):
    """Colorbar −120…+120 m with discrete depositional-bin colors."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    sm, bounds = _depositional_scalar_mappable()
    if isinstance(ax, (list, tuple, np.ndarray)):
        host = list(np.ravel(ax))[-1]
    else:
        host = ax
    cax = inset_axes(
        host,
        width="4.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=host.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(sm, cax=cax, label=label, spacing="proportional")
    cbar.set_ticks(list(bounds))
    cbar.ax.tick_params(labelsize="x-small")
    return cbar


def plot_compare_maps(
    xa,
    ya,
    elev_a,
    ed_a,
    xb,
    yb,
    elev_b,
    ed_b,
    *,
    cells_a=None,
    cells_b=None,
    title_a="Run A: goSPL SPL + linear diffusion",
    title_b="Run B: goSPL SPL + Landlab NLD + HyLands",
    out_png: Optional[Path] = None,
):
    """2×2: elev A|B (no mesh); erodep A|B with mesh overlay."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)

    def _sc(ax, x, y, v, cells, cmap, title, label, mesh: bool, elev: bool):
        if v is None:
            ax.set_title(title + " (missing)")
            ax.axis("off")
            return
        if elev:
            sc = _map_elev(ax, x, y, v, cells=cells, cmap=cmap)
        else:
            sc = _map_field(ax, x, y, v, cells=cells, cmap=cmap)
        if mesh:
            _overlay_mesh(ax, x, y, cells)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        _add_cbar(fig, sc, ax, label)

    _sc(axs[0, 0], xa, ya, elev_a, cells_a, "terrain", title_a, _CBAR_ELEV, False, True)
    _sc(axs[0, 1], xb, yb, elev_b, cells_b, "terrain", title_b, _CBAR_ELEV, False, True)
    _sc(axs[1, 0], xa, ya, ed_a, cells_a, "RdBu", title_a, _CBAR_CUMED, True, False)
    _sc(axs[1, 1], xb, yb, ed_b, cells_b, "RdBu", title_b, _CBAR_CUMED, True, False)

    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def plot_mid_transect(
    x: np.ndarray,
    elev: np.ndarray,
    *,
    strata_z: Optional[np.ndarray] = None,
    label: str = "",
    ax=None,
):
    """Simple elevation line; optional stacked strata elevations along transect."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    ax.plot(x, elev, "k-", lw=1.5, label=label or "surface")
    if strata_z is not None:
        for i in range(strata_z.shape[0]):
            ax.plot(x, strata_z[i], lw=0.6, alpha=0.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("elevation (m)")
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def _draw_strata_section(
    ax,
    x_km: np.ndarray,
    elev: np.ndarray,
    strata,
    *,
    title: str,
    strat_h=None,
    layer_dt_yr: Optional[float] = None,
    line_interval_yr: float = _LINE_INTERVAL_YR,
    show_paleo_legend: bool = False,
) -> None:
    """
    Mid-transect stratal section (whole-transect x-range).

    Prefer stacking positive ``stratH`` under ``elev``. Fills colored by paleo
    elev / water-depth bins when ``strata`` (stratZ) is present. Black lines only
    on ``line_interval_yr`` milestones (default 0.5 Myr); thicker t=0 basement
    and current surface. Sea level at elev=0 (light blue, behind).
    """
    elev = np.asarray(elev, dtype=float).ravel()
    x_km = np.asarray(x_km, dtype=float).ravel()

    # sea level behind everything
    ax.axhline(
        0.0,
        color=_SEA_LEVEL_COLOR,
        alpha=_SEA_LEVEL_ALPHA,
        lw=1.2,
        zorder=0,
        solid_capstyle="butt",
    )

    drew = False
    if strat_h is not None:
        H = np.asarray(strat_h, dtype=float)
        if H.ndim == 2 and H.shape[0] == elev.size:
            interfaces, thick = sediment_interfaces_from_thickness(elev, H)
            n_lay = thick.shape[0]
            # paleo elev per layer: strata as (n_lay, n) or (n, n_lay)
            paleo = None
            if strata is not None:
                S = np.asarray(strata, dtype=float)
                if S.ndim == 2 and S.shape[0] == n_lay and S.shape[1] == elev.size:
                    paleo = S
                elif S.ndim == 2 and S.shape[1] == n_lay and S.shape[0] == elev.size:
                    paleo = S.T
            milestones = _milestone_layer_indices(
                n_lay, layer_dt_yr=layer_dt_yr, line_interval_yr=line_interval_yr
            )
            y0 = float(np.nanmin(interfaces[0]))
            pad = max(50.0, 0.05 * (float(np.nanmax(elev)) - y0 + 1.0))
            ax.fill_between(
                x_km,
                y0 - pad,
                interfaces[0],
                facecolor="gainsboro",
                linewidth=0,
                alpha=0.95,
                zorder=0.5,
            )
            for i in range(n_lay):
                prow = paleo[i] if paleo is not None else None
                _fill_layer_by_paleo(
                    ax,
                    x_km,
                    interfaces[i],
                    interfaces[i + 1],
                    thick[i],
                    prow,
                    i,
                )
                if i in milestones:
                    ax.plot(x_km, interfaces[i + 1], "-", color="k", lw=0.7, zorder=2)
            # t=0 basement + current surface (thicker)
            ax.plot(x_km, interfaces[0], "-", color="k", lw=1.2, zorder=3)
            ax.plot(x_km, elev, "k-", lw=1.2, label="surface", zorder=3)
            drew = True
        elif H.ndim == 2 and H.shape[1] == elev.size:
            return _draw_strata_section(
                ax,
                x_km,
                elev,
                strata,
                title=title,
                strat_h=H.T,
                layer_dt_yr=layer_dt_yr,
                line_interval_yr=line_interval_yr,
                show_paleo_legend=show_paleo_legend,
            )

    if not drew and strata is not None and strata.ndim == 2 and strata.shape[0] >= 2:
        nlay = strata.shape[0]
        milestones = _milestone_layer_indices(
            nlay - 1, layer_dt_yr=layer_dt_yr, line_interval_yr=line_interval_yr
        )
        for i in range(nlay - 1):
            thick_row = np.abs(strata[i + 1] - strata[i])
            paleo_row = 0.5 * (strata[i] + strata[i + 1])
            _fill_layer_by_paleo(
                ax, x_km, strata[i], strata[i + 1], thick_row, paleo_row, i
            )
            if i in milestones:
                ax.plot(x_km, strata[i + 1], "-", color="k", lw=0.7, zorder=2)
        ax.plot(x_km, strata[0], "-", color="k", lw=1.2, zorder=3)
        ax.plot(x_km, strata[-1], "-", color="k", lw=1.2, zorder=3)

    if not drew:
        ax.plot(x_km, elev, "k-", lw=1.2, label="surface", zorder=3)

    ax.set_xlim(float(x_km.min()), float(x_km.max()))
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)
    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    if show_paleo_legend:
        _add_paleo_legend(ax)


def plot_frame_maps(
    run_a: Path,
    run_b: Path,
    step: int,
    *,
    elev_clim: Optional[Tuple[float, float]] = None,
    erodep_clim: Optional[Tuple[float, float]] = None,
    out_png: Optional[Path] = None,
    title: Optional[str] = None,
):
    """
    One time: 2×2 maps — row1 elev A|B (no mesh), row2 erodep A|B (mesh overlay).
    """
    import matplotlib.pyplot as plt

    a = load_gospl_step(run_a, step)
    b = load_gospl_step(run_b, step)
    if elev_clim is None:
        elev_clim = (
            min(np.nanmin(a["elev"]), np.nanmin(b["elev"])),
            max(np.nanmax(a["elev"]), np.nanmax(b["elev"])),
        )
    if erodep_clim is None:
        erodep_clim = (
            min(np.nanmin(a["erodep"]), np.nanmin(b["erodep"])),
            max(np.nanmax(a["erodep"]), np.nanmax(b["erodep"])),
        )
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    # ``title`` kept for API compat; figure-level time header removed (panel titles carry t).
    _ = title

    specs = (
        (axs[0, 0], a, a["elev"], "terrain", elev_clim, "A", "elev", _CBAR_ELEV, False),
        (axs[0, 1], b, b["elev"], "terrain", elev_clim, "B", "elev", _CBAR_ELEV, False),
        (axs[1, 0], a, a["erodep"], "RdBu", erodep_clim, "A", "cumed", _CBAR_CUMED, True),
        (axs[1, 1], b, b["erodep"], "RdBu", erodep_clim, "B", "cumed", _CBAR_CUMED, True),
    )
    for ax, d, v, cmap, clim, run, kind, label, mesh in specs:
        if kind == "elev":
            sc = _map_elev(
                ax,
                d["x"],
                d["y"],
                v,
                cells=d.get("cells"),
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
            )
        else:
            sc = _map_field(
                ax,
                d["x"],
                d["y"],
                v,
                cells=d.get("cells"),
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
            )
        if mesh:
            _overlay_mesh(ax, d["x"], d["y"], d.get("cells"))
        ax.set_aspect("equal")
        ax.set_title(_panel_title(run, kind, d["time_yr"], step))
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        _add_cbar(fig, sc, ax, label)

    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    return fig


def _progress_png_name(out_root: Path, time_yr: Optional[float], step: int) -> Path:
    if time_yr is not None and time_yr >= 0.1e6:
        tag = f"{time_yr / 1e6:.1f}".replace(".", "p") + "Myr"
    elif time_yr is not None:
        tag = f"{time_yr / 1e3:.0f}kyr"
    else:
        tag = f"step{step}"
    return Path(out_root) / f"compare_maps_progress_{tag}.png"


def plot_progress_maps(
    run_a: Path,
    run_b: Path,
    steps: Sequence[int],
    *,
    out_png: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """
    One 2×2 elev|erodep+mesh figure per progress time.

    Writes ``compare_maps_progress_<time>.png`` under ``out_dir`` (or parent of
    ``out_png``). Also writes a stacked multi-time PNG to ``out_png`` when given
    (legacy path name ``compare_maps_progress.png``).
    """
    import matplotlib.pyplot as plt

    steps = list(steps)
    elev_clim = _shared_clim(run_a, run_b, steps, "elev")
    erodep_clim = _shared_clim(run_a, run_b, steps, "erodep")
    root = Path(out_dir) if out_dir is not None else (
        Path(out_png).parent if out_png is not None else Path(".")
    )
    written: List[Path] = []
    for step in steps:
        a = load_gospl_step(run_a, step)
        path = _progress_png_name(root, a["time_yr"], step)
        fig = plot_frame_maps(
            run_a,
            run_b,
            step,
            elev_clim=elev_clim,
            erodep_clim=erodep_clim,
            out_png=path,
        )
        plt.close(fig)
        written.append(path)

    # stacked overview: nrows = n_times, each "row block" is elev|elev then we
    # only keep per-time files as primary; optional combined elev-only skipped.
    if out_png is not None and written:
        # copy first path list note: write a simple contact-sheet of the 3 PNGs
        # is heavy; instead re-draw as n_times * 2 rows × 2 cols
        n = len(steps)
        fig, axs = plt.subplots(n * 2, 2, figsize=(20, 7.6 * n), constrained_layout=True)
        if n * 2 == 1:
            axs = np.asarray([axs])
        for i, step in enumerate(steps):
            a = load_gospl_step(run_a, step)
            b = load_gospl_step(run_b, step)
            for col, d, run in ((0, a, "A"), (1, b, "B")):
                ax_e = axs[2 * i, col]
                ax_d = axs[2 * i + 1, col]
                sc_e = _map_elev(
                    ax_e, d["x"], d["y"], d["elev"], cells=d.get("cells"),
                    cmap="terrain", vmin=elev_clim[0], vmax=elev_clim[1],
                )
                sc_d = _map_field(
                    ax_d, d["x"], d["y"], d["erodep"], cells=d.get("cells"),
                    cmap="RdBu", vmin=erodep_clim[0], vmax=erodep_clim[1],
                )
                _overlay_mesh(ax_d, d["x"], d["y"], d.get("cells"))
                for ax in (ax_e, ax_d):
                    ax.set_aspect("equal")
                    ax.set_xlabel("x (km)")
                    ax.set_ylabel("y (km)")
                ax_e.set_title(_panel_title(run, "elev", d["time_yr"], step))
                ax_d.set_title(_panel_title(run, "cumed", d["time_yr"], step))
                _add_cbar(fig, sc_e, ax_e, _CBAR_ELEV)
                _add_cbar(fig, sc_d, ax_d, _CBAR_CUMED)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        written.append(Path(out_png))
    return written


def plot_progress_sections(
    run_a: Path,
    run_b: Path,
    steps: Sequence[int],
    *,
    out_png: Optional[Path] = None,
    layer_dt_yr: Optional[float] = 5000.0,
    line_interval_yr: float = _LINE_INTERVAL_YR,
):
    """
    One row per time step: mid-transect stratal section for A | B.

    Whole-transect x-range; sediment ribbons from positive ``stratH`` when present.
    """
    import matplotlib.pyplot as plt

    nrows = len(steps)
    fig, axs = plt.subplots(nrows, 2, figsize=(22, 6.0 * nrows), constrained_layout=True)
    if nrows == 1:
        axs = np.asarray([axs])

    for i, step in enumerate(steps):
        a = load_gospl_step(run_a, step)
        b = load_gospl_step(run_b, step)
        for j, (ax, d, run) in enumerate(
            (
                (axs[i, 0], a, "A"),
                (axs[i, 1], b, "B"),
            )
        ):
            ids = mid_transect_indices(d["x"], d["y"])
            x_km = d["x"][ids] / 1000.0
            elev = d["elev"][ids]
            strata = None
            strat_h = None
            if d["strata"] is not None:
                strata = d["strata"][ids, :].T
            if d.get("strat_h") is not None:
                strat_h = d["strat_h"][ids, :]
            _draw_strata_section(
                ax,
                x_km,
                elev,
                strata,
                title=_panel_title(run, "elev", d["time_yr"], step),
                strat_h=strat_h,
                layer_dt_yr=layer_dt_yr,
                line_interval_yr=line_interval_yr,
                show_paleo_legend=(j == 0),
            )
            xmid = 0.5 * (d["x"].min() + d["x"].max()) / 1000.0
            ax.axvline(xmid, color="gray", ls="--", lw=0.7)

    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def _shared_clim(run_a: Path, run_b: Path, steps: Sequence[int], field: str):
    """Min/max of ``field`` over steps×runs via running reduce (no big concat)."""
    lo = np.inf
    hi = -np.inf
    for step in steps:
        for run in (run_a, run_b):
            d = load_gospl_step(run, step)
            v = np.asarray(d[field], dtype=float).ravel()
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            lo = min(lo, float(v.min()))
            hi = max(hi, float(v.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    return lo, hi


def _update_minmax(lo: float, hi: float, arr: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(arr, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return lo, hi
    return min(lo, float(v.min())), max(hi, float(v.max()))


def _shared_combo_limits(
    run_a: Path,
    run_b: Path,
    steps: Sequence[int],
    *,
    need_elev: bool = True,
    need_erodep: bool = True,
    need_zlim: bool = True,
    show_progress: bool = True,
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
]:
    """
    One HDF5 pass for whichever of elev clim / erodep clim / section zlim are needed.

    Pass ``need_*=False`` to skip that panel's min/max (caller already fixed it).
    Returns ``None`` for any skipped limit.
    """
    from tqdm.auto import tqdm

    if not (need_elev or need_erodep or need_zlim):
        return None, None, None

    elev_lo, elev_hi = np.inf, -np.inf
    ed_lo, ed_hi = np.inf, -np.inf
    z_lo, z_hi = np.inf, -np.inf
    ids_cache: Dict[str, np.ndarray] = {}
    step_iter: Sequence[int] = list(steps)
    if show_progress:
        which = "+".join(
            n
            for n, on in (
                ("elev", need_elev),
                ("erodep", need_erodep),
                ("zlim", need_zlim),
            )
            if on
        )
        step_iter = tqdm(step_iter, desc=f"shared scales ({which})", unit="step")

    for step in step_iter:
        for run in (run_a, run_b):
            try:
                d = load_gospl_step(run, step)
            except FileNotFoundError:
                continue
            elev = np.asarray(d["elev"], dtype=float).ravel()
            if need_elev:
                elev_lo, elev_hi = _update_minmax(elev_lo, elev_hi, elev)
            if need_erodep:
                erodep = np.asarray(d["erodep"], dtype=float).ravel()
                ed_lo, ed_hi = _update_minmax(ed_lo, ed_hi, erodep)

            if need_zlim:
                key = str(run)
                if key not in ids_cache:
                    ids_cache[key] = mid_transect_indices(d["x"], d["y"])
                ids = ids_cache[key]
                z_lo, z_hi = _update_minmax(z_lo, z_hi, elev[ids])

                strat_h = d.get("strat_h")
                if strat_h is not None:
                    H = np.asarray(strat_h, dtype=float)
                    if H.ndim == 2 and H.shape[0] == elev.size:
                        interfaces, _ = sediment_interfaces_from_thickness(
                            elev[ids], H[ids, :]
                        )
                        z_lo, z_hi = _update_minmax(z_lo, z_hi, interfaces[0])
                elif d.get("strata") is not None:
                    S = np.asarray(d["strata"], dtype=float)
                    if S.ndim == 2 and S.shape[0] == elev.size:
                        z_lo, z_hi = _update_minmax(z_lo, z_hi, S[ids, :])
                    elif S.ndim == 2 and S.shape[1] == elev.size:
                        z_lo, z_hi = _update_minmax(z_lo, z_hi, S[:, ids])

    elev_out: Optional[Tuple[float, float]] = None
    erodep_out: Optional[Tuple[float, float]] = None
    zlim_out: Optional[Tuple[float, float]] = None
    if need_elev:
        if not np.isfinite(elev_lo):
            elev_lo, elev_hi = 0.0, 1.0
        elev_out = (elev_lo, elev_hi)
    if need_erodep:
        if not np.isfinite(ed_lo):
            ed_lo, ed_hi = 0.0, 1.0
        erodep_out = (ed_lo, ed_hi)
    if need_zlim:
        if not np.isfinite(z_lo):
            if elev_out is not None:
                z_lo, z_hi = elev_out
            else:
                z_lo, z_hi = 0.0, 1.0
        pad = max(50.0, 0.05 * (z_hi - z_lo + 1.0))
        zlim_out = (z_lo - pad, z_hi + pad)
    return elev_out, erodep_out, zlim_out


def animate_maps_gif(
    run_a: Path,
    run_b: Path,
    *,
    out_gif: Path,
    fps: float = 4.5,
    steps: Optional[Sequence[int]] = None,
) -> Path:
    """GIF: each frame is 2×2 elev A|B (no mesh) + erodep A|B (mesh overlay)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib as mpl

    run_a, run_b = Path(run_a), Path(run_b)
    steps = shared_existing_steps(run_a, run_b, steps)
    if not steps:
        raise FileNotFoundError(f"no shared HDF5 steps for GIF under {run_a} / {run_b}")
    elev_clim = _shared_clim(run_a, run_b, steps, "elev")
    erodep_clim = _shared_clim(run_a, run_b, steps, "erodep")

    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    sm_e = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=elev_clim[0], vmax=elev_clim[1]), cmap="terrain"
    )
    sm_d = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=erodep_clim[0], vmax=erodep_clim[1]), cmap="RdBu"
    )
    sm_e.set_array([])
    sm_d.set_array([])
    _add_cbar(fig, sm_e, axs[0, 1], _CBAR_ELEV)
    _add_cbar(fig, sm_d, axs[1, 1], _CBAR_CUMED)
    last = {"a": None, "b": None, "step": steps[0]}

    def _update(i):
        step = steps[i]
        pa = run_a / "h5" / f"gospl.{step}.p0.h5"
        pb = run_b / "h5" / f"gospl.{step}.p0.h5"
        if pa.is_file() and pb.is_file():
            last["a"] = load_gospl_step(run_a, step)
            last["b"] = load_gospl_step(run_b, step)
            last["step"] = step
        elif last["a"] is None:
            return axs.ravel()
        a, b = last["a"], last["b"]
        for ax in axs.ravel():
            ax.clear()
            ax.set_aspect("equal")
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
        for ax, d, field, cmap, clim, mesh, run, kind in (
            (axs[0, 0], a, "elev", "terrain", elev_clim, False, "A", "elev"),
            (axs[0, 1], b, "elev", "terrain", elev_clim, False, "B", "elev"),
            (axs[1, 0], a, "erodep", "RdBu", erodep_clim, True, "A", "cumed"),
            (axs[1, 1], b, "erodep", "RdBu", erodep_clim, True, "B", "cumed"),
        ):
            if kind == "elev":
                _map_elev(
                    ax, d["x"], d["y"], d[field], cells=d.get("cells"),
                    cmap=cmap, vmin=clim[0], vmax=clim[1],
                )
            else:
                _map_field(
                    ax, d["x"], d["y"], d[field], cells=d.get("cells"),
                    cmap=cmap, vmin=clim[0], vmax=clim[1],
                )
            if mesh:
                _overlay_mesh(ax, d["x"], d["y"], d.get("cells"))
            ax.set_title(_panel_title(run, kind, d["time_yr"], last["step"]))
        return axs.ravel()

    _update(0)
    anim = FuncAnimation(fig, _update, frames=len(steps), interval=1000.0 / fps, blit=False)
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_gif


def animate_sections_gif(
    run_a: Path,
    run_b: Path,
    *,
    out_gif: Path,
    fps: float = 4.5,
    steps: Optional[Sequence[int]] = None,
    layer_dt_yr: Optional[float] = 5000.0,
    line_interval_yr: float = _LINE_INTERVAL_YR,
) -> Path:
    """GIF of mid-transect stratal sections for every output step."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    run_a, run_b = Path(run_a), Path(run_b)
    steps = shared_existing_steps(run_a, run_b, steps)
    if not steps:
        raise FileNotFoundError(f"no shared HDF5 steps for GIF under {run_a} / {run_b}")

    fig, axs = plt.subplots(1, 2, figsize=(22, 7.0), constrained_layout=True)
    last = {"a": None, "b": None, "step": steps[0]}

    def _frame(i):
        for ax in axs:
            ax.clear()
        step = steps[i]
        pa = run_a / "h5" / f"gospl.{step}.p0.h5"
        pb = run_b / "h5" / f"gospl.{step}.p0.h5"
        if pa.is_file() and pb.is_file():
            last["a"] = load_gospl_step(run_a, step)
            last["b"] = load_gospl_step(run_b, step)
            last["step"] = step
        elif last["a"] is None:
            return axs
        a, b = last["a"], last["b"]
        for j, (ax, d, run) in enumerate(
            (
                (axs[0], a, "A"),
                (axs[1], b, "B"),
            )
        ):
            ids = mid_transect_indices(d["x"], d["y"])
            x_km = d["x"][ids] / 1000.0
            elev = d["elev"][ids]
            strata = None
            strat_h = None
            if d["strata"] is not None:
                strata = d["strata"][ids, :].T
            if d.get("strat_h") is not None:
                strat_h = d["strat_h"][ids, :]
            _draw_strata_section(
                ax,
                x_km,
                elev,
                strata,
                title=_panel_title(run, "elev", d["time_yr"], last["step"]),
                strat_h=strat_h,
                layer_dt_yr=layer_dt_yr,
                line_interval_yr=line_interval_yr,
                show_paleo_legend=(j == 0),
            )
            xmid = 0.5 * (d["x"].min() + d["x"].max()) / 1000.0
            ax.axvline(xmid, color="gray", ls="--", lw=0.7)
        return axs

    _frame(0)
    anim = FuncAnimation(fig, _frame, frames=len(steps), interval=1000.0 / fps, blit=False)
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_gif


def _combo_frames_dir(out_root: Path) -> Path:
    """Frame PNGs live under ``out_root/frames``; GIF stays under ``out_root``."""
    return Path(out_root) / "frames"


def _combo_frame_png_name(out_root: Path, step: int) -> Path:
    return _combo_frames_dir(out_root) / f"compare_combo_frame_{int(step):05d}.png"


def _shared_section_zlim(
    run_a: Path,
    run_b: Path,
    steps: Sequence[int],
) -> Tuple[float, float]:
    """Shared mid-transect elev range (incl. basement) over steps, with pad."""
    _, _, zlim = _shared_combo_limits(
        run_a,
        run_b,
        steps,
        need_elev=False,
        need_erodep=False,
        need_zlim=True,
        show_progress=False,
    )
    assert zlim is not None
    return zlim


def plot_frame_combo(
    run_a: Path,
    run_b: Path,
    step: int,
    *,
    elev_clim: Optional[Tuple[float, float]] = None,
    erodep_clim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    out_png: Optional[Path] = None,
    dpi: float = 200.0,
    layer_dt_yr: Optional[float] = 5000.0,
    line_interval_yr: float = _LINE_INTERVAL_YR,
    show_mesh: bool = True,
):
    """
    One time: 3×2 combo — elev A|B, cumED A|B (± mesh), mid-transect A|B.

    Tight column / colorbar spacing (Nature-like). Equal row heights; dedicated
    right-hand colorbar column. ``show_mesh`` toggles triangle overlay on the
    cumED row only. Saves at ``dpi`` when ``out_png`` is set.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    a = load_gospl_step(run_a, step)
    b = load_gospl_step(run_b, step)
    if elev_clim is None:
        elev_clim = (
            min(float(np.nanmin(a["elev"])), float(np.nanmin(b["elev"]))),
            max(float(np.nanmax(a["elev"])), float(np.nanmax(b["elev"]))),
        )
    if erodep_clim is None:
        erodep_clim = (
            min(float(np.nanmin(a["erodep"])), float(np.nanmin(b["erodep"]))),
            max(float(np.nanmax(a["erodep"])), float(np.nanmax(b["erodep"]))),
        )

    # Compact figure: two map columns + thin cbar (avoid wide constrained gaps)
    fig = plt.figure(figsize=(11.2, 10.4))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 0.028],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.10,
        hspace=0.28,
        left=0.07,
        right=0.93,
        top=0.93,
        bottom=0.06,
    )
    axs = np.empty((3, 2), dtype=object)
    caxes = []
    for r in range(3):
        for c in range(2):
            axs[r, c] = fig.add_subplot(gs[r, c])
        caxes.append(fig.add_subplot(gs[r, 2]))

    # Match map geographic aspect so all three rows share the same panel height.
    x0 = float(min(np.nanmin(a["x"]), np.nanmin(b["x"]))) / 1000.0
    x1 = float(max(np.nanmax(a["x"]), np.nanmax(b["x"]))) / 1000.0
    y0 = float(min(np.nanmin(a["y"]), np.nanmin(b["y"]))) / 1000.0
    y1 = float(max(np.nanmax(a["y"]), np.nanmax(b["y"]))) / 1000.0
    map_box_aspect = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)

    # Row 0 elev, row 1 cumED — colorbars in right column
    for row, field, cmap, clim, label, mesh, kind in (
        (0, "elev", "terrain", elev_clim, _CBAR_ELEV, False, "elev"),
        (1, "erodep", "RdBu", erodep_clim, _CBAR_CUMED, bool(show_mesh), "cumed"),
    ):
        for col, d, run in ((0, a, "A"), (1, b, "B")):
            ax = axs[row, col]
            v = d[field]
            if kind == "elev":
                _map_elev(
                    ax,
                    d["x"],
                    d["y"],
                    v,
                    cells=d.get("cells"),
                    cmap=cmap,
                    vmin=clim[0],
                    vmax=clim[1],
                )
            else:
                _map_field(
                    ax,
                    d["x"],
                    d["y"],
                    v,
                    cells=d.get("cells"),
                    cmap=cmap,
                    vmin=clim[0],
                    vmax=clim[1],
                )
            if mesh:
                _overlay_mesh(ax, d["x"], d["y"], d.get("cells"))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            # Box aspect alone (no set_aspect equal) so row heights match sections.
            ax.set_box_aspect(map_box_aspect)
            ax.set_title(_panel_title(run, kind, d["time_yr"], step), fontsize=9)
            ax.set_xlabel("x (km)", fontsize=8)
            ax.set_ylabel("y (km)", fontsize=8)
            ax.tick_params(labelsize=7)
        cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        sm = mpl.cm.ScalarMappable(
            norm=Normalize(vmin=clim[0], vmax=clim[1]), cmap=cmap_obj
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=caxes[row], label=label)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(label, fontsize=8)

    for j, (ax, d, run) in enumerate(
        (
            (axs[2, 0], a, "A"),
            (axs[2, 1], b, "B"),
        )
    ):
        ids = mid_transect_indices(d["x"], d["y"])
        x_km = d["x"][ids] / 1000.0
        elev = d["elev"][ids]
        strata = None
        strat_h = None
        if d["strata"] is not None:
            strata = d["strata"][ids, :].T
        if d.get("strat_h") is not None:
            strat_h = d["strat_h"][ids, :]
        _draw_strata_section(
            ax,
            x_km,
            elev,
            strata,
            title=_panel_title(run, "elev", d["time_yr"], step),
            strat_h=strat_h,
            layer_dt_yr=layer_dt_yr,
            line_interval_yr=line_interval_yr,
            show_paleo_legend=False,
        )
        xmid = 0.5 * (d["x"].min() + d["x"].max()) / 1000.0
        ax.axvline(xmid, color="gray", ls="--", lw=0.7)
        if zlim is not None:
            ax.set_ylim(zlim[0], zlim[1])
        ax.set_box_aspect(map_box_aspect)
        ax.set_title(ax.get_title(), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(ax.get_xlabel() or "x (km)", fontsize=8)
        ax.set_ylabel(ax.get_ylabel() or "z (m)", fontsize=8)

    # Depositional elev/depth: linear −120…+120 m; colors follow bin edges.
    sm_p, bounds = _depositional_scalar_mappable()
    cbar_p = fig.colorbar(
        sm_p,
        cax=caxes[2],
        label="Depositional elev / depth (m)",
        spacing="proportional",
    )
    cbar_p.set_ticks(list(bounds))
    cbar_p.ax.tick_params(labelsize=7)
    cbar_p.set_label("Depositional elev / depth (m)", fontsize=8)

    # Pin colorbars flush to the right of the B column (tight Nature-like gap)
    fig.canvas.draw()
    for row, cax in enumerate(caxes):
        pos = axs[row, 1].get_position()
        cax.set_position([pos.x1 + 0.008, pos.y0, 0.016, pos.height])

    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi)
    return fig


def _write_one_combo_frame(payload: dict) -> str:
    """Joblib worker: write one combo PNG (Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_a = Path(payload["run_a"])
    run_b = Path(payload["run_b"])
    step = int(payload["step"])
    if (
        resolve_gospl_h5(run_a, step, repair=True) is None
        or resolve_gospl_h5(run_b, step, repair=True) is None
    ):
        return ""

    fig = plot_frame_combo(
        run_a,
        run_b,
        step,
        elev_clim=tuple(payload["elev_clim"]),
        erodep_clim=tuple(payload["erodep_clim"]),
        zlim=tuple(payload["zlim"]) if payload.get("zlim") is not None else None,
        out_png=Path(payload["out_png"]),
        dpi=float(payload["dpi"]),
        layer_dt_yr=payload.get("layer_dt_yr"),
        line_interval_yr=float(payload.get("line_interval_yr", _LINE_INTERVAL_YR)),
        show_mesh=bool(payload.get("show_mesh", True)),
    )
    plt.close(fig)
    return str(payload["out_png"])


def write_combo_frames(
    out_root: Path,
    *,
    run_a_name: str = "runA_gospl_spl",
    run_b_name: str = "runB_space_hylands",
    gif_interval_yr: float = 10_000.0,
    progress_times_yr: Optional[Sequence[float]] = (0.5e6, 1.5e6, 3.0e6),
    dpi: float = 200.0,
    n_jobs: int = 10,
    layer_dt_yr: Optional[float] = 5000.0,
    line_interval_yr: float = _LINE_INTERVAL_YR,
    elev_clim: Optional[Tuple[float, float]] = None,
    erodep_clim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    show_mesh: bool = True,
) -> dict:
    """
    Write 3-row combo PNGs for every GIF-interval step (existing HDF5 only).

    Parallelized with ``joblib`` (``n_jobs``, processes). Returns paths including
    ``progress_frames`` nearest ``progress_times_yr``.

    Color / section limits: pass ``(vmin, vmax)`` to fix a scale for all frames;
    ``None`` (default) auto-shares min/max across A|B and all GIF steps. Applies
    to elev maps, cumED maps, and section y-limits independently.

    ``show_mesh``: overlay goSPL triangles on the cumED row (default True).
    """
    from joblib import Parallel, delayed

    out_root = Path(out_root)
    run_a = latest_run_dir(out_root / run_a_name)
    run_b = latest_run_dir(out_root / run_b_name)
    common = shared_existing_steps(run_a, run_b)
    if not common:
        raise RuntimeError(f"no shared output steps between {run_a} and {run_b}")
    gif_steps = pick_steps_every_interval(
        run_a, common, float(gif_interval_yr), dt_yr=layer_dt_yr
    )
    gif_steps = shared_existing_steps(run_a, run_b, gif_steps)
    if not gif_steps:
        raise RuntimeError("no GIF steps available on disk")

    need_elev = elev_clim is None
    need_erodep = erodep_clim is None
    need_zlim = zlim is None
    if need_elev or need_erodep or need_zlim:
        print(
            "combo scales: "
            f"elev={'auto' if need_elev else tuple(map(float, elev_clim))}, "
            f"erodep={'auto' if need_erodep else tuple(map(float, erodep_clim))}, "
            f"zlim={'auto' if need_zlim else tuple(map(float, zlim))}"
        )
        auto_elev, auto_erodep, auto_zlim = _shared_combo_limits(
            run_a,
            run_b,
            gif_steps,
            need_elev=need_elev,
            need_erodep=need_erodep,
            need_zlim=need_zlim,
        )
        if need_elev:
            elev_clim = auto_elev
        if need_erodep:
            erodep_clim = auto_erodep
        if need_zlim:
            zlim = auto_zlim
    else:
        print(
            "combo scales: all fixed "
            f"(elev={tuple(map(float, elev_clim))}, "
            f"erodep={tuple(map(float, erodep_clim))}, "
            f"zlim={tuple(map(float, zlim))}) — skip min/max scan"
        )
    elev_clim = (float(elev_clim[0]), float(elev_clim[1]))
    erodep_clim = (float(erodep_clim[0]), float(erodep_clim[1]))
    zlim = (float(zlim[0]), float(zlim[1]))

    frames_dir = _combo_frames_dir(out_root)
    frames_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    combo_frames: List[Path] = []
    for step in gif_steps:
        out_png = _combo_frame_png_name(out_root, step)
        combo_frames.append(out_png)
        payloads.append(
            {
                "run_a": str(run_a),
                "run_b": str(run_b),
                "step": int(step),
                "out_png": str(out_png),
                "elev_clim": [float(elev_clim[0]), float(elev_clim[1])],
                "erodep_clim": [float(erodep_clim[0]), float(erodep_clim[1])],
                "zlim": [float(zlim[0]), float(zlim[1])],
                "dpi": float(dpi),
                "layer_dt_yr": layer_dt_yr,
                "line_interval_yr": float(line_interval_yr),
                "show_mesh": bool(show_mesh),
            }
        )

    n_jobs = max(1, int(n_jobs))
    from tqdm.auto import tqdm
    import warnings

    # return_as=generator so tqdm advances on completed frames (not just dispatch)
    # Loky may emit a benign "worker stopped…" UserWarning under memory pressure;
    # keep n_jobs as requested and hide that noise when frames still complete.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*worker stopped while some jobs were given to the executor.*",
            category=UserWarning,
        )
        done = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
            delayed(_write_one_combo_frame)(p) for p in payloads
        )
        for _ in tqdm(done, total=len(payloads), desc="combo frames", unit="frame"):
            pass

    progress_frames: List[Path] = []
    if progress_times_yr:
        prog_steps = pick_steps_nearest_times(
            run_a, gif_steps, progress_times_yr, dt_yr=layer_dt_yr
        )
        prog_steps = shared_existing_steps(run_a, run_b, prog_steps)
        for step in prog_steps:
            p = _combo_frame_png_name(out_root, step)
            if p.is_file():
                progress_frames.append(p)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "steps_gif": gif_steps,
        "combo_frames": combo_frames,
        "progress_frames": progress_frames,
        "elev_clim": elev_clim,
        "erodep_clim": erodep_clim,
        "zlim": zlim,
        "out_root": out_root,
        "frames_dir": frames_dir,
    }


def list_combo_frame_pngs(out_root: Path) -> List[Path]:
    """Ordered ``compare_combo_frame_XXXXX.png`` under ``out_root/frames``."""
    frames_dir = _combo_frames_dir(out_root)
    if frames_dir.is_dir():
        paths = sorted(frames_dir.glob("compare_combo_frame_*.png"))
        if paths:
            return paths
    # Fallback: legacy flat layout under ``out_root``
    return sorted(Path(out_root).glob("compare_combo_frame_*.png"))


def _load_frame_rgb_array(path: Path) -> np.ndarray:
    """Load one PNG as contiguous RGB uint8; pad to even size for yuv420p."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return np.ascontiguousarray(arr)


def stitch_combo_mp4(
    frame_paths: Sequence[Path],
    out_mp4: Path,
    *,
    fps: float = 16.2,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """
    Build H.264 MP4 from full-resolution combo PNGs (no downscale).

    Streams frames through ffmpeg (``imageio-ffmpeg`` binary): CRF compression,
    ``yuv420p`` + ``+faststart`` for compatible playback and accurate fps timing.
    Default ``fps=16.2`` (1.2× the prior 13.5).
    """
    import subprocess

    import imageio_ffmpeg
    from tqdm.auto import tqdm

    frames = [Path(p) for p in frame_paths]
    if not frames:
        raise FileNotFoundError("no combo frame PNGs to stitch")
    for p in frames:
        if not p.is_file():
            raise FileNotFoundError(p)

    first = _load_frame_rgb_array(frames[0])
    height, width = first.shape[:2]
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    fps = float(fps)
    crf = int(crf)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        f"{fps:.6g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        str(preset),
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    print(
        f"encoding MP4 {width}×{height} × {len(frames)} frames "
        f"@ {fps:g} fps (crf={crf}, preset={preset}) → {out_mp4}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(first.tobytes())
        for p in tqdm(frames[1:], total=len(frames) - 1, desc="MP4 encode", unit="frame"):
            arr = _load_frame_rgb_array(p)
            if arr.shape[0] != height or arr.shape[1] != width:
                raise ValueError(
                    f"frame size {arr.shape[1]}x{arr.shape[0]} != {width}x{height} ({p})"
                )
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr is not None else b""
        rc = proc.wait()
    except Exception:
        proc.kill()
        raise
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {rc}): {err.decode('utf-8', errors='replace')[-2000:]}"
        )
    return out_mp4


def stitch_combo_gif(
    frame_paths: Sequence[Path],
    out_gif: Path,
    *,
    fps: float = 16.2,
    **kwargs,
) -> Path:
    """Deprecated alias: writes MP4 (``.mp4``) instead of GIF; no downscale."""
    out = Path(out_gif)
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
    return stitch_combo_mp4(frame_paths, out, fps=fps, **kwargs)


def make_progress_and_gifs(
    out_root: Path,
    *,
    run_a_name: str = "runA_gospl_spl",
    run_b_name: str = "runB_space_hylands",
    n_progress: int = 3,
    progress_times_yr: Optional[Sequence[float]] = (0.5e6, 2.0e6, 4.0e6),
    make_gifs: bool = True,
    gif_interval_yr: Optional[float] = 20_000.0,
    gif_fps: float = 4.5,
    layer_dt_yr: Optional[float] = 5000.0,
    close_figures: bool = True,
    include_maps: bool = True,
    include_sections: bool = True,
) -> dict:
    """
    Build progress PNGs (+ optional GIFs) under ``out_root``.

    Maps: one 2×2 (elev | erodep+mesh) PNG per progress time, plus stacked
    ``compare_maps_progress.png``. No separate erodep-only progress/GIF.
    Use ``include_maps=False`` / ``include_sections=False`` to split notebook
    cells; ``close_figures=True`` clears leftover pyplot state.
    """
    import matplotlib.pyplot as plt

    out_root = Path(out_root)
    run_a = latest_run_dir(out_root / run_a_name)
    run_b = latest_run_dir(out_root / run_b_name)
    common = shared_existing_steps(run_a, run_b)
    if not common:
        raise RuntimeError(f"no shared output steps between {run_a} and {run_b}")
    if progress_times_yr:
        progress = pick_steps_nearest_times(
            run_a, common, progress_times_yr, dt_yr=layer_dt_yr
        )
    else:
        progress = pick_progress_steps(common, n=n_progress)
    progress = shared_existing_steps(run_a, run_b, progress)
    if not progress:
        raise RuntimeError("progress steps missing on disk (file missing from a cloud-synced folder?)")

    if gif_interval_yr is None:
        gif_steps = list(common)
    else:
        gif_steps = pick_steps_every_interval(
            run_a, common, float(gif_interval_yr), dt_yr=layer_dt_yr
        )
    gif_steps = shared_existing_steps(run_a, run_b, gif_steps)

    maps_progress_png = out_root / "compare_maps_progress.png"
    paths: Dict[str, object] = {
        "run_a": run_a,
        "run_b": run_b,
        "steps_all": common,
        "steps_progress": progress,
        "steps_gif": gif_steps,
        "maps_progress": maps_progress_png,
        "maps_progress_frames": [],
        "sections_progress": out_root / "compare_sections_progress.png",
        "maps_gif": out_root / "compare_maps.gif",
        "sections_gif": out_root / "compare_sections.gif",
    }
    if include_maps:
        frame_paths = plot_progress_maps(
            run_a, run_b, progress, out_png=maps_progress_png, out_dir=out_root
        )
        paths["maps_progress_frames"] = [
            p for p in frame_paths if p != maps_progress_png
        ]
        if make_gifs:
            gif_now = shared_existing_steps(run_a, run_b, gif_steps)
            paths["steps_gif"] = gif_now
            animate_maps_gif(
                run_a, run_b, out_gif=paths["maps_gif"], steps=gif_now, fps=gif_fps
            )
        if close_figures:
            plt.close("all")
    if include_sections:
        progress_now = shared_existing_steps(run_a, run_b, progress)
        fig_sec = plot_progress_sections(
            run_a,
            run_b,
            progress_now,
            out_png=paths["sections_progress"],
            layer_dt_yr=layer_dt_yr,
        )
        if make_gifs:
            gif_now = shared_existing_steps(run_a, run_b, gif_steps)
            paths["steps_gif"] = gif_now
            animate_sections_gif(
                run_a,
                run_b,
                out_gif=paths["sections_gif"],
                steps=gif_now,
                fps=gif_fps,
                layer_dt_yr=layer_dt_yr,
            )
        if close_figures:
            plt.close(fig_sec)
            plt.close("all")
    return paths

