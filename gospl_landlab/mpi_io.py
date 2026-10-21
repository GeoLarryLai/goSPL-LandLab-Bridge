"""MPI gather/scatter helpers for rank-0 Landlab."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def get_comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def gather_field(local: np.ndarray, counts: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Gather variable-length local arrays to rank 0.

    If ``counts`` is None, assumes equal length on all ranks (Allgather).
    """
    comm = get_comm()
    local = np.ascontiguousarray(local, dtype=float)
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size == 1:
        return local.copy()

    if counts is None:
        # equal chunks
        n = local.size
        all_n = comm.allgather(n)
        if len(set(all_n)) != 1:
            raise ValueError("local sizes differ; pass counts for Allgatherv")
        buf = np.empty(n * size, dtype=float)
        comm.Allgather([local, n], [buf, n])
        return buf if rank == 0 else None

    counts = np.asarray(counts, dtype=int)
    displ = np.insert(np.cumsum(counts), 0, 0)[:-1]
    total = int(counts.sum())
    recv = np.empty(total, dtype=float) if rank == 0 else None
    comm.Gatherv(local, [recv, counts, displ, None] if rank == 0 else None, root=0)
    return recv


def scatter_field(global_on_root: Optional[np.ndarray], local_size: int) -> np.ndarray:
    """Broadcast full array from rank 0, then each rank takes its slice if equal partition.

    For goSPL we typically set the *global* PETSc vector from a full array on every
    rank after bcast (simpler than matching DM ownership). Caller decides how to apply.
    """
    comm = get_comm()
    if comm.Get_size() == 1:
        return np.asarray(global_on_root, dtype=float).copy()

    if comm.Get_rank() == 0:
        arr = np.asarray(global_on_root, dtype=float).ravel()
        n = arr.size
    else:
        arr = None
        n = None
    n = comm.bcast(n, root=0)
    if comm.Get_rank() != 0:
        arr = np.empty(n, dtype=float)
    comm.Bcast(arr, root=0)
    return arr


def warn_if_large(n_global: int, threshold: int, rank: int = 0) -> None:
    if rank == 0 and n_global >= threshold:
        print(
            f"[gospl_landlab] WARNING: gathering {n_global} nodes to rank 0 "
            f"(threshold={threshold}). Memory/time may dominate.",
            flush=True,
        )
