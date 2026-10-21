"""Landlab grid mirrors of a goSPL mesh (irregular 1:1 and regular raster remap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MirrorState:
    grid: object
    """Landlab ModelGrid (Voronoi irregular or Raster)."""

    gospl_to_ll: np.ndarray
    """Landlab node id for each goSPL node."""

    ll_to_gospl: np.ndarray
    """Inverse map (Landlab node → gathered goSPL index); -1 if unused."""

    shape: Optional[Tuple[int, int]] = None
    """(ny, nx) when raster."""

    dx: Optional[float] = None

    gospl_xy: Optional[np.ndarray] = None
    """``(n, 2)`` goSPL world ``(x, y)`` in gathered order."""

    xmin: float = 0.0
    ymin: float = 0.0

    use_interp: bool = True
    """If True, elev push/pull use cached barycentric weights (+ nearest fill)."""

    kind: str = "raster"
    """``raster`` (DDTD + landslide) or ``irregular`` (optional legacy)."""

    dtype: np.dtype = np.dtype(np.float64)
    """Landlab field dtype for this mirror (float32 or float64)."""

    # Query xy (float64) for push (raster nodes) / pull (goSPL nodes)
    _push_xy_ll64: Optional[np.ndarray] = None
    _pull_pts64: Optional[np.ndarray] = None
    _pull_xy_ll64: Optional[np.ndarray] = None

    # Barycentric weight tables (built once; same math as LinearND + nearest)
    _push_idx: Optional[np.ndarray] = None  # (n_raster, 3) source vertex ids
    _push_w: Optional[np.ndarray] = None  # (n_raster, 3) weights
    _push_near: Optional[np.ndarray] = None  # (n_raster,) nearest src for exterior
    _push_miss: Optional[np.ndarray] = None  # (n_raster,) bool
    _pull_idx: Optional[np.ndarray] = None
    _pull_w: Optional[np.ndarray] = None
    _pull_near: Optional[np.ndarray] = None
    _pull_miss: Optional[np.ndarray] = None


def _weight_chunk_size(n: int) -> int:
    """Chunk size for weight-table build tqdm."""
    if n <= 0:
        return 1
    return max(1000, min(8000, n // 20 or n))


def _build_barycentric_table(
    src_xy: np.ndarray,
    query_xy: np.ndarray,
    *,
    desc: str,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute LinearND-equivalent barycentric indices/weights + nearest fill.

    Returns ``(idx, weights, near_idx, miss)`` with shapes ``(n,3)``, ``(n,3)``,
    ``(n,)``, ``(n,)``.
    """
    from scipy.spatial import Delaunay, cKDTree

    src = np.asarray(src_xy, dtype=np.float64)
    q = np.asarray(query_xy, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError("query_xy must be (n, 2)")
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_xy must be (n, 2)")

    n = int(q.shape[0])
    d = 2
    tri = Delaunay(src)
    tree = cKDTree(src)

    idx = np.zeros((n, 3), dtype=np.intp)
    w = np.zeros((n, 3), dtype=np.float64)
    near = np.zeros(n, dtype=np.intp)
    miss = np.ones(n, dtype=bool)

    chunk = _weight_chunk_size(n)
    n_chunks = max(1, (n + chunk - 1) // chunk)
    iterator = range(0, n, chunk)

    if show_progress:
        from tqdm.auto import tqdm

        from .progress import _tqdm_kwargs

        iterator = tqdm(iterator, **_tqdm_kwargs(desc, total=n_chunks))

    for a in iterator:
        b = min(a + chunk, n)
        qq = q[a:b]
        s = tri.find_simplex(qq)
        ok = s >= 0
        miss[a:b] = ~ok
        if np.any(ok):
            simp = s[ok]
            T = tri.transform[simp, :d]
            delta = qq[ok] - tri.transform[simp, d]
            bary = np.einsum("nij,nj->ni", T, delta)
            ww = np.empty((simp.shape[0], 3), dtype=np.float64)
            ww[:, :d] = bary
            ww[:, d] = 1.0 - bary.sum(axis=1)
            rows = np.flatnonzero(ok) + a
            idx[rows] = tri.simplices[simp]
            w[rows] = ww
        if np.any(~ok):
            _, nn = tree.query(qq[~ok], k=1)
            rows_m = np.flatnonzero(~ok) + a
            near[rows_m] = np.asarray(nn, dtype=np.intp)

    return idx, w, near, miss


def _eval_barycentric(
    idx: np.ndarray,
    weights: np.ndarray,
    near: np.ndarray,
    miss: np.ndarray,
    values: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply cached barycentric weights (NumPy); nearest where ``miss``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = int(idx.shape[0])
    if out is None:
        out = np.empty(n, dtype=np.float64)
    elif out.shape != (n,):
        raise ValueError("out buffer length mismatch")
    out[:] = (
        weights[:, 0] * v[idx[:, 0]]
        + weights[:, 1] * v[idx[:, 1]]
        + weights[:, 2] * v[idx[:, 2]]
    )
    if np.any(miss):
        out[miss] = v[near[miss]]
    return out


def push_weights_ready(mirror: MirrorState) -> bool:
    return (
        mirror._push_idx is not None
        and mirror._push_w is not None
        and mirror._push_near is not None
        and mirror._push_miss is not None
    )


def pull_weights_ready(mirror: MirrorState) -> bool:
    return (
        mirror._pull_idx is not None
        and mirror._pull_w is not None
        and mirror._pull_near is not None
        and mirror._pull_miss is not None
    )


def build_push_weight_cache(
    mirror: MirrorState, *, show_progress: bool = True
) -> None:
    """Build goSPL→raster barycentric weights (own tqdm when ``show_progress``)."""
    if not mirror.use_interp or mirror.gospl_xy is None:
        return
    if push_weights_ready(mirror):
        return
    mirror._push_xy_ll64 = np.asarray(_raster_xy(mirror), dtype=np.float64)
    src = np.asarray(mirror.gospl_xy, dtype=np.float64)
    idx, w, near, miss = _build_barycentric_table(
        src,
        mirror._push_xy_ll64,
        desc="Remap weights goSPL→raster",
        show_progress=show_progress,
    )
    mirror._push_idx = idx
    mirror._push_w = w
    mirror._push_near = near
    mirror._push_miss = miss


def build_pull_weight_cache(
    mirror: MirrorState, *, show_progress: bool = True
) -> None:
    """Build raster→goSPL barycentric weights (own tqdm when ``show_progress``)."""
    if not mirror.use_interp or mirror.gospl_xy is None:
        return
    if pull_weights_ready(mirror):
        return
    xy_ll = np.asarray(_raster_xy(mirror), dtype=np.float64)
    mirror._pull_xy_ll64 = xy_ll
    mirror._pull_pts64 = np.asarray(mirror.gospl_xy, dtype=np.float64)
    idx, w, near, miss = _build_barycentric_table(
        xy_ll,
        mirror._pull_pts64,
        desc="Remap weights raster→goSPL",
        show_progress=show_progress,
    )
    mirror._pull_idx = idx
    mirror._pull_w = w
    mirror._pull_near = near
    mirror._pull_miss = miss


def ensure_remap_weight_caches(
    mirror: MirrorState, *, show_progress: bool = True
) -> None:
    """Ensure both push and pull weight tables exist (separate tqdm each)."""
    build_push_weight_cache(mirror, show_progress=show_progress)
    build_pull_weight_cache(mirror, show_progress=show_progress)


def push_elevation(mirror: MirrorState, z_gospl: np.ndarray) -> None:
    """Push goSPL elevations onto the mirror (direct map or interpolate)."""
    dt = mirror.dtype
    z = np.asarray(z_gospl, dtype=dt).ravel()
    if not mirror.use_interp or mirror.gospl_xy is None:
        if z.size != mirror.gospl_to_ll.size:
            raise ValueError("z_gospl length must match goSPL node count")
        elev = mirror.grid.at_node["topographic__elevation"]
        elev[mirror.gospl_to_ll] = z
        return
    if z.size != mirror.gospl_xy.shape[0]:
        raise ValueError("z_gospl length must match gospl_xy")
    if not push_weights_ready(mirror):
        build_push_weight_cache(mirror, show_progress=False)
    elev = mirror.grid.at_node["topographic__elevation"]
    # Prefer in-place float64 write (raster path); else eval then cast-assign
    if (
        np.dtype(elev.dtype) == np.dtype(np.float64)
        and elev.size == mirror._push_idx.shape[0]
    ):
        _eval_barycentric(
            mirror._push_idx,
            mirror._push_w,
            mirror._push_near,
            mirror._push_miss,
            z,
            out=np.asarray(elev, dtype=np.float64).ravel(),
        )
    else:
        z_ll = _eval_barycentric(
            mirror._push_idx,
            mirror._push_w,
            mirror._push_near,
            mirror._push_miss,
            z,
        )
        elev[:] = np.asarray(z_ll, dtype=dt)


def pull_elevation(mirror: MirrorState) -> np.ndarray:
    """Sample mirror elevations back to goSPL node order (float64 for goSPL)."""
    z_ll = np.asarray(mirror.grid.at_node["topographic__elevation"], dtype=np.float64)
    if not mirror.use_interp or mirror.gospl_xy is None:
        return np.asarray(z_ll[mirror.gospl_to_ll], dtype=np.float64).copy()
    if not pull_weights_ready(mirror):
        build_pull_weight_cache(mirror, show_progress=False)
    return _eval_barycentric(
        mirror._pull_idx,
        mirror._pull_w,
        mirror._pull_near,
        mirror._pull_miss,
        z_ll,
    )


def pull_field_to_gospl(
    mirror: MirrorState,
    values_ll: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map a Landlab-node field onto goSPL node order (float64 for goSPL)."""
    v = np.asarray(values_ll, dtype=np.float64).ravel()
    if not mirror.use_interp or mirror.gospl_xy is None:
        if v.size != mirror.grid.number_of_nodes:
            raise ValueError("field length must match irregular mirror nodes")
        result = np.asarray(v[mirror.gospl_to_ll], dtype=np.float64)
        if out is not None:
            out[:] = result
            return out
        return result.copy()
    if v.size != mirror.grid.number_of_nodes:
        raise ValueError("field length must match raster mirror nodes")
    if not pull_weights_ready(mirror):
        build_pull_weight_cache(mirror, show_progress=False)
    return _eval_barycentric(
        mirror._pull_idx,
        mirror._pull_w,
        mirror._pull_near,
        mirror._pull_miss,
        v,
        out=out,
    )


def resolve_landlab_dtype(dtype) -> np.dtype:
    """Map config/name to a numpy floating dtype (float32 or float64)."""
    if isinstance(dtype, np.dtype):
        dt = dtype
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dt = np.dtype(dtype)
    else:
        key = str(dtype).lower().strip()
        if key in {"float32", "f4", "single", "f32"}:
            dt = np.dtype(np.float32)
        elif key in {"float64", "f8", "double", "float", "f64"}:
            dt = np.dtype(np.float64)
        else:
            raise ValueError(f"landlab_dtype must be float32 or float64, got {dtype!r}")
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"landlab_dtype must be float32 or float64, got {dt}")
    return dt


def _estimate_dx_nx_ny(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dx: Optional[float],
    nx: Optional[int],
    ny: Optional[int],
) -> Tuple[float, int, int, float, float]:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if dx is None:
        from scipy.spatial import cKDTree

        pts = np.column_stack([x, y])
        n = pts.shape[0]
        if n > 4000:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=4000, replace=False)
            sample = pts[idx]
        else:
            sample = pts
        tree = cKDTree(pts)
        d, _ = tree.query(sample, k=2)
        dx = float(np.median(d[:, 1]))
        if not np.isfinite(dx) or dx <= 0:
            raise ValueError("could not infer dx from point spacing")
    if nx is None:
        nx = int(round((xmax - xmin) / dx)) + 1
    if ny is None:
        ny = int(round((ymax - ymin) / dx)) + 1
    nx = max(int(nx), 2)
    ny = max(int(ny), 2)
    return float(dx), nx, ny, xmin, ymin


def _seed_soil_bedrock(
    mg, z: np.ndarray, *, soil0: float = 1.0, dtype=np.float64
) -> None:
    dt = resolve_landlab_dtype(dtype)
    elev = mg.add_zeros("topographic__elevation", at="node", dtype=dt)
    elev[:] = np.asarray(z, dtype=dt)
    soil = mg.add_zeros("soil__depth", at="node", dtype=dt)
    soil[:] = dt.type(soil0)
    br = mg.add_zeros("bedrock__elevation", at="node", dtype=dt)
    br[:] = elev - soil
    if "soil_production__rate" not in mg.at_node:
        mg.add_zeros("soil_production__rate", at="node", dtype=dt)


def build_irregular_mirror(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dtype=np.float64,
) -> MirrorState:
    """
    Build a ``VoronoiDelaunayGrid`` from goSPL ``(x,y)``.

    Coords are owned contiguous copies at ``dtype`` (avoids F-VORO-BYTES).
    Landlab may reorder nodes (esp. float32); ``gospl_to_ll`` maps goSPL → LL.
    """
    from landlab import VoronoiDelaunayGrid
    from scipy.spatial import cKDTree

    dt = resolve_landlab_dtype(dtype)
    # Own contiguous copies. goSPL/PETSc views can share a buffer whose .base
    # chain ends in bytes; Landlab Graph.freeze() then raises on .flags.
    x = np.array(x, dtype=dt, copy=True).ravel()
    y = np.array(y, dtype=dt, copy=True).ravel()
    z = np.array(z, dtype=dt, copy=True).ravel()
    if x.size != y.size or x.size != z.size:
        raise ValueError("x, y, z length mismatch")

    mg = VoronoiDelaunayGrid(x, y)
    if mg.number_of_nodes != x.size:
        raise RuntimeError(
            f"VoronoiDelaunayGrid node count {mg.number_of_nodes} != goSPL {x.size}"
        )

    pts = np.column_stack([x, y])
    xy_ll = np.column_stack(
        [np.asarray(mg.x_of_node, dtype=dt), np.asarray(mg.y_of_node, dtype=dt)]
    )
    _, gospl_to_ll = cKDTree(xy_ll).query(pts, k=1)
    gospl_to_ll = np.asarray(gospl_to_ll, dtype=int)
    if gospl_to_ll.size != len(np.unique(gospl_to_ll)):
        raise RuntimeError("VoronoiDelaunayGrid node map is not 1:1 with goSPL")
    d_chk = np.hypot(xy_ll[gospl_to_ll, 0] - x, xy_ll[gospl_to_ll, 1] - y)
    tol = 1e-2 if dt == np.dtype(np.float32) else 1e-6
    if float(np.max(d_chk)) > tol:
        raise RuntimeError(
            "VoronoiDelaunayGrid node map xy mismatch "
            f"(max {float(np.max(d_chk))} m)"
        )

    z_ll = np.empty(mg.number_of_nodes, dtype=dt)
    z_ll[gospl_to_ll] = z
    _seed_soil_bedrock(mg, z_ll, soil0=1.0, dtype=dt)

    ll_to_gospl = np.full(mg.number_of_nodes, -1, dtype=int)
    ll_to_gospl[gospl_to_ll] = np.arange(x.size, dtype=int)
    dx, _, _, xmin, ymin = _estimate_dx_nx_ny(x, y, dx=None, nx=None, ny=None)

    return MirrorState(
        grid=mg,
        gospl_to_ll=gospl_to_ll,
        ll_to_gospl=ll_to_gospl,
        shape=None,
        dx=dx,
        gospl_xy=pts,
        xmin=float(xmin),
        ymin=float(ymin),
        use_interp=False,
        kind="irregular",
        dtype=dt,
    )


def build_raster_mirror(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dx: Optional[float] = None,
    ny: Optional[int] = None,
    nx: Optional[int] = None,
    scale: float = 1.0,
    dtype=np.float64,
) -> MirrorState:
    """
    Build a regular ``RasterModelGrid`` and scatter-map from irregular goSPL nodes.

    ``scale`` refines the raster: ``dx_raster = dx_mesh / scale`` (default 1.0).
    Initial elevation uses LinearND once; runtime push/pull use cached
    barycentric weights built by ``build_push_weight_cache`` /
    ``build_pull_weight_cache``.
    """
    from landlab import RasterModelGrid
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    dt = resolve_landlab_dtype(dtype)
    x = np.array(x, dtype=dt, copy=True).ravel()
    y = np.array(y, dtype=dt, copy=True).ravel()
    z = np.array(z, dtype=dt, copy=True).ravel()
    if x.size != y.size or x.size != z.size:
        raise ValueError("x, y, z length mismatch")

    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be > 0")

    dx_mesh, _, _, xmin, ymin = _estimate_dx_nx_ny(x, y, dx=dx, nx=None, ny=None)
    dx_r = float(dx_mesh) / scale if dx is None else float(dx) / scale
    dx_r, nx_r, ny_r, xmin, ymin = _estimate_dx_nx_ny(
        x,
        y,
        dx=dx_r,
        nx=None if scale != 1.0 or nx is None else nx,
        ny=None if scale != 1.0 or ny is None else ny,
    )
    if scale == 1.0 and nx is not None:
        nx_r = max(int(nx), 2)
    if scale == 1.0 and ny is not None:
        ny_r = max(int(ny), 2)

    mg = RasterModelGrid(
        (ny_r, nx_r), xy_spacing=dx_r, xy_of_lower_left=(xmin, ymin)
    )
    xy_ll = np.column_stack(
        [np.asarray(mg.x_of_node, dtype=dt), np.asarray(mg.y_of_node, dtype=dt)]
    )

    pts = np.column_stack([x, y])
    # LinearND works in float64; store result as Landlab dtype
    lin = LinearNDInterpolator(
        np.asarray(pts, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    near = NearestNDInterpolator(
        np.asarray(pts, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    xy64 = np.asarray(xy_ll, dtype=np.float64)
    z_ll = lin(xy64)
    miss = ~np.isfinite(z_ll)
    if np.any(miss):
        z_ll[miss] = near(xy64[miss])
    z_ll = np.asarray(z_ll, dtype=dt)

    from scipy.spatial import cKDTree

    tree = cKDTree(xy_ll)
    _, gospl_to_ll = tree.query(pts, k=1)
    gospl_to_ll = np.asarray(gospl_to_ll, dtype=int)

    ll_to_gospl = np.full(mg.number_of_nodes, -1, dtype=int)
    for g, ll in enumerate(gospl_to_ll):
        ll_to_gospl[ll] = g

    _seed_soil_bedrock(mg, z_ll, soil0=1.0, dtype=dt)

    if "cell_area_at_node" not in mg.at_node:
        area = mg.add_zeros("cell_area_at_node", at="node", dtype=dt)
        area[:] = dt.type(dx_r * dx_r)

    return MirrorState(
        grid=mg,
        gospl_to_ll=gospl_to_ll,
        ll_to_gospl=ll_to_gospl,
        shape=(ny_r, nx_r),
        dx=dx_r,
        gospl_xy=pts,
        xmin=xmin,
        ymin=ymin,
        use_interp=True,
        kind="raster",
        dtype=dt,
    )


def _raster_xy(mirror: MirrorState) -> np.ndarray:
    mg = mirror.grid
    dt = mirror.dtype
    return np.column_stack(
        [np.asarray(mg.x_of_node, dtype=dt), np.asarray(mg.y_of_node, dtype=dt)]
    )
