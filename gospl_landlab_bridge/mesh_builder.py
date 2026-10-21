"""Build a flat goSPL ``.npz`` mesh + uplift/subsidence map for demos.

Default demo mesh: lattice with interior xy jitter (irregular triangles) plus
optional elevation noise. Landlab HyLands still uses a regular raster mirror
with interpolate remapping (see ``mirror.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay


def build_regular_flat_mesh(
    out_prefix: Path,
    *,
    nx: int = 61,
    ny: int = 41,
    dx: float = 1000.0,
    z0: float = 0.0,
    uplift_left_m_per_yr: float = 0.001,
    mountain_width_m: float = 30_000.0,
    sigma0_m_per_yr: float = 0.001,
    alpha_per_m: Optional[float] = None,
    xy_jitter_frac: float = 0.2,
    elev_noise_m: float = 1.0,
    seed: int = 0,
) -> Tuple[Path, Path]:
    """
    Write ``{out_prefix}.npz`` (v,c,n,z) and ``{out_prefix}_tect.npz`` (z rock-uplift rates).

    Starts on a square lattice, then jitters **interior** ``x,y`` by up to
    ``xy_jitter_frac * dx`` (domain edges stay on the box). Optional
    ``elev_noise_m`` adds uniform noise in ``[-elev_noise_m, +elev_noise_m]``
    with ``seed`` (shared A/B when callers use the same seed).

    Tectonic map (no live flexure), after Duller et al. (2010) / Wild et al. (2025 ESurf):

    - **Mountain** (``x <= mountain_width_m``, default left 30 km): constant uplift
      ``U`` only (default ``0.001`` m/yr = 1 mm/yr). The exponential is **not**
      applied on the mountain.
    - **Basin** (``x > mountain_width_m`` only):
      ``σ(x) = σ₀ · exp(−α · (x − x_front))`` with ``σ₀`` at the front
      (default 1 mm/yr). Stored as **negative** uplift rates for goSPL ``upsub``.

    If ``alpha_per_m`` is None, use ``α = 2 / L_B`` (e-folding over half the
    basin length).

    Name kept for callers; output is irregular when ``xy_jitter_frac > 0``.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    xx, yy = np.meshgrid(xs, ys)
    x = xx.ravel().astype(float)
    y = yy.ravel().astype(float)
    z = np.full(nx * ny, float(z0), dtype=float)

    rng = np.random.default_rng(int(seed))
    # Interior: not on domain boundary rows/cols
    col = np.tile(np.arange(nx), ny)
    row = np.repeat(np.arange(ny), nx)
    interior = (col > 0) & (col < nx - 1) & (row > 0) & (row < ny - 1)
    amp = float(xy_jitter_frac) * float(dx)
    if amp > 0.0 and np.any(interior):
        x[interior] += rng.uniform(-amp, amp, size=int(interior.sum()))
        y[interior] += rng.uniform(-amp, amp, size=int(interior.sum()))
        # keep inside the original bounding box
        x = np.clip(x, float(xs.min()), float(xs.max()))
        y = np.clip(y, float(ys.min()), float(ys.max()))

    if float(elev_noise_m) > 0.0:
        z = z + rng.uniform(-float(elev_noise_m), float(elev_noise_m), size=z.size)

    v = np.column_stack([x, y, np.zeros(nx * ny)])

    tri = Delaunay(v[:, :2])
    c = tri.simplices.astype(np.int64)

    # neighbour matrix (up to 8) — goSPL expects (npoints, 8)
    n = np.full((v.shape[0], 8), -1, dtype=np.int64)
    from collections import defaultdict

    adj = defaultdict(set)
    for a, b, d in c:
        adj[a].update((b, d))
        adj[b].update((a, d))
        adj[d].update((a, b))
    for i, nbrs in adj.items():
        lst = sorted(nbrs)[:8]
        n[i, : len(lst)] = lst

    mesh_path = Path(str(out_prefix) + ".npz")
    np.savez_compressed(
        mesh_path,
        v=v,
        c=c,
        n=n,
        z=z,
        nx=nx,
        ny=ny,
        dx=float(dx),
        seed=int(seed),
        xy_jitter_frac=float(xy_jitter_frac),
        elev_noise_m=float(elev_noise_m),
    )

    # --- rock uplift / subsidence rates (goSPL upsub: + uplift, − subsidence) ---
    # Left  [0, x_front]:  constant U only (never add/subtract σ here).
    # Right (x_front, L]:  σ(x) = σ₀·exp(−α·(x − x_front)), σ₀ at the front.
    # Use *lattice* x for tectonics so mountain/basin partition is clean.
    x_front = float(mountain_width_m)
    x_lattice = xx.ravel()
    if x_front <= float(xs.min()) or x_front >= float(xs.max()):
        raise ValueError(
            f"mountain_width_m={mountain_width_m} must lie inside the domain "
            f"[{xs.min()}, {xs.max()}]"
        )
    L_B = float(xs.max() - x_front)
    alpha = float(2.0 / L_B) if alpha_per_m is None else float(alpha_per_m)

    u = np.full(x_lattice.shape, float(uplift_left_m_per_yr), dtype=float)
    basin = x_lattice > x_front
    x_b = x_lattice[basin] - float(np.min(x_lattice[basin]))
    u[basin] = -float(sigma0_m_per_yr) * np.exp(-alpha * x_b)

    if np.any(u[~basin] != float(uplift_left_m_per_yr)):
        raise RuntimeError("mountain uplift was contaminated by the basin profile")
    if np.any(u[basin] > 0):
        raise RuntimeError("basin nodes have positive uplift; expected subsidence ≤ 0")

    tect_path = Path(str(out_prefix) + "_tect.npz")
    np.savez_compressed(tect_path, z=u)

    return mesh_path, tect_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "examples" / "inputs"
    build_regular_flat_mesh(root / "compare_flat")
    print("wrote", root / "compare_flat.npz")
