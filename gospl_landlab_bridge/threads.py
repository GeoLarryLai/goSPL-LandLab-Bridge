"""Optional BLAS / OpenMP thread caps for numpy-heavy Landlab work."""

from __future__ import annotations

import os
from typing import Optional

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Last value from ``set_blas_threads`` (notebook ``LANDLAB_BLAS_THREADS``).
_LAST_N: Optional[int] = None


def set_blas_threads(n: int) -> int:
    """
    Cap BLAS/OpenMP threads for vectorized numpy / scipy work.

    Sets common env vars and ``GOSPL_LANDLAB_BLAS_THREADS`` (used by NumPy
    remap / Landlab). Best effect if called at process start (before heavy
    imports). Does **not** parallelize Landlab's Python landslide scar loop.

    Returns the clamped thread count used.
    """
    global _LAST_N
    n = max(1, int(n))
    s = str(n)
    _LAST_N = n
    os.environ["GOSPL_LANDLAB_BLAS_THREADS"] = s
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = s
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(n)
    except Exception:
        pass
    return n


def blas_threads_from_env(default: Optional[int] = None) -> Optional[int]:
    """
    Thread count for BLAS / Landlab NumPy work.

    Prefers last ``set_blas_threads`` value, then ``GOSPL_LANDLAB_BLAS_THREADS``,
    then ``OMP_NUM_THREADS``, else ``default``.
    """
    if _LAST_N is not None:
        return max(1, int(_LAST_N))
    for key in ("GOSPL_LANDLAB_BLAS_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return max(1, int(raw))
    return default
