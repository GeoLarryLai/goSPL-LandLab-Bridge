"""Runtime monkeypatches for known goSPL bugs (we do not edit goSPL's install tree).

Adapted from goSPL (GPL-3.0). goSPL copyright remains with its authors.
"""

from __future__ import annotations

_APPLIED = False


def apply_gospl_patches() -> None:
    """Idempotent. Call after ``import gospl`` (e.g. from ``load_model``)."""
    global _APPLIED
    if _APPLIED:
        return

    _patch_fill_from_edges()
    _APPLIED = True


def _patch_fill_from_edges() -> None:
    """
    goSPL 2026.7.14 ``PITFill._fillFromEdges`` sizes ``ggraph`` as
    ``max(target_label) + 2``. When the highest pit ID appears only as a
    *source* edge, indexing ``ggraph[uniq]`` raises IndexError
    (e.g. ``index 8 is out of bounds for axis 0 with size 8``).

    Fix: size from ``max(source, target) + 2``. Same fill result otherwise.
    """
    import numpy as np
    import pandas as pd
    from gospl.flow.pitfilling import PITFill, fill_edges

    def _fillFromEdges(self, mgraph):
        cgraph = pd.DataFrame(
            mgraph, columns=["source", "target", "elev", "spill", "rank"]
        )
        cgraph = cgraph.sort_values("elev")
        cgraph = cgraph.drop_duplicates(["source", "target"], keep="first")
        cgraph = cgraph.values

        # FIX (upstream uses cgraph[:, 1].max() only):
        nb_global = int(cgraph[:, :2].max()) + 2
        src = cgraph[:, 0].astype(np.int64)
        tgt = cgraph[:, 1].astype(np.int64)
        uniq = np.unique(np.concatenate((src, tgt)))
        if uniq[0] != 0:
            uniq = np.concatenate((np.array([0], dtype=uniq.dtype), uniq))
        nlab = len(uniq)
        ccgraph = cgraph.copy()
        ccgraph[:, 0] = np.searchsorted(uniq, src)
        ccgraph[:, 1] = np.searchsorted(uniq, tgt)
        c12 = ccgraph[:, :2].astype(int).ravel()
        cmax = np.max(np.bincount(c12)) + 1

        self.profiler.start("flow_pit_edges")
        elev, rank, nodes, spillID = fill_edges(nlab + 1, ccgraph, cmax)
        self.profiler.stop("flow_pit_edges")

        ggraph = -np.ones((nb_global, 5))
        ggraph[:, 0] = 1.0e8
        ggraph[uniq, 0] = elev[:nlab]
        ggraph[uniq, 1] = nodes[:nlab]
        ggraph[uniq, 2] = rank[:nlab]
        gspill = spillID[:nlab].astype(np.int64)
        valid = (gspill >= 0) & (gspill < nlab)
        gspill[valid] = uniq[gspill[valid]]
        ggraph[uniq, 3] = gspill
        if self.memclear:
            del elev, nodes, rank, spillID
            del src, tgt, uniq, c12, gspill, ccgraph

        return ggraph

    PITFill._fillFromEdges = _fillFromEdges
