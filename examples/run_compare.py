#!/usr/bin/env python
"""Run comparison experiments (A: goSPL SPL+LD; B: goSPL SPL + Landlab NLD/HyLands)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _boot_gospl():
    from gospl_landlab_bridge import install_vtk_shim

    install_vtk_shim(force=True)
    from gospl.model import Model

    return Model


def run_a(yml: Path) -> None:
    Model = _boot_gospl()
    m = Model(str(yml), verbose=True)
    m.runProcesses()
    m.destroy()


def run_b(yml: Path, bridge_yml: Path) -> None:
    import yaml

    from gospl_landlab_bridge import BridgeConfig, BridgeRunner

    Model = _boot_gospl()
    with open(bridge_yml) as f:
        cfg = BridgeConfig.from_dict(yaml.safe_load(f))
    m = Model(str(yml), verbose=True)
    BridgeRunner(m, cfg).run()
    m.destroy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=["A", "B", "both"])
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args(argv)
    root = args.root
    ya = root / "examples" / "runA_gospl_spl.yml"
    yb = root / "examples" / "runB_space_hylands.yml"
    br = root / "examples" / "bridge_runB.yml"
    if args.which in ("A", "both"):
        print("=== Run A: goSPL SPL + linear diffusion ===", flush=True)
        run_a(ya)
    if args.which in ("B", "both"):
        print("=== Run B: goSPL SPL + Landlab NLD/HyLands ===", flush=True)
        run_b(yb, br)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
