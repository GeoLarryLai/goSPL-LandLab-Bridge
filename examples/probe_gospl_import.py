"""Probe whether goSPL Model imports after VTK shim."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gospl_landlab import install_vtk_shim

install_vtk_shim(force=True)
print("shim ok", flush=True)
from gospl.model import Model

print("Model OK", Model, flush=True)
