#!/usr/bin/env python
"""Make progress maps/sections (3 steps) + GIFs for all goSPL output steps."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gospl_landlab.postprocess import (  # noqa: E402
    latest_run_dir,
    list_gospl_steps,
    load_gospl_step,
    make_progress_and_gifs,
    mid_transect_indices,
    plot_compare_maps,
)


def main():
    out = ROOT / "examples" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    paths = make_progress_and_gifs(out)
    print("A:", paths["run_a"])
    print("B:", paths["run_b"])
    print("progress steps:", paths["steps_progress"])
    print("all steps:", paths["steps_all"])
    for k in (
        "maps_progress",
        "maps_progress_frames",
        "sections_progress",
        "maps_gif",
        "sections_gif",
    ):
        print("wrote", paths[k])

    # also keep end-step 2×2 elev/erodep snapshot
    a_dir, b_dir = paths["run_a"], paths["run_b"]
    step = paths["steps_all"][-1]
    a = load_gospl_step(a_dir, step)
    b = load_gospl_step(b_dir, step)
    plot_compare_maps(
        a["x"],
        a["y"],
        a["elev"],
        a["erodep"],
        b["x"],
        b["y"],
        b["elev"],
        b["erodep"],
        cells_a=a.get("cells"),
        cells_b=b.get("cells"),
        out_png=out / "compare_maps.png",
    )
    ids_a = mid_transect_indices(a["x"], a["y"])
    ids_b = mid_transect_indices(b["x"], b["y"])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(a["x"][ids_a] / 1e3, a["elev"][ids_a], label="A: goSPL SPL + diffusion")
    ax.plot(b["x"][ids_b] / 1e3, b["elev"][ids_b], label="B: SPL + NLD + HyLands")
    ax.axvline(0.5 * (a["x"].min() + a["x"].max()) / 1e3, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(f"Mid-transect topography (final step {step})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "compare_transect.png", dpi=150)
    print("wrote", out / "compare_maps.png", out / "compare_transect.png")


if __name__ == "__main__":
    main()
