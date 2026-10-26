"""goSPL–Landlab bridge: load Landlab components onto goSPL runs without forking either repo."""

from .vtk_shim import install_vtk_shim
from .config import BridgeConfig
from .runner import BridgeRunner
from .capability import probe_component, CAPABILITY_TABLE
from .progress import load_model, format_run_done, destroy_model, landlab_substeps_for_dt
from .threads import set_blas_threads

__all__ = [
    "install_vtk_shim",
    "BridgeConfig",
    "BridgeRunner",
    "probe_component",
    "CAPABILITY_TABLE",
    "load_model",
    "format_run_done",
    "destroy_model",
    "landlab_substeps_for_dt",
    "set_blas_threads",
]

__version__ = "0.0.4"
