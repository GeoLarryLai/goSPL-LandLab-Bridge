"""Bridge configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _is_off(backend: str) -> bool:
    return backend.lower() in {"off", "none", "disabled", ""}


@dataclass
class SlotSpec:
    """One process slot: either goSPL builtin, a Landlab component, or ``off``."""

    backend: str
    """``gospl.spl``, ``gospl.linear_diffusion``, ``landlab.Space``, ``off``, …"""

    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_off(self) -> bool:
        return _is_off(self.backend)

    @property
    def is_landlab(self) -> bool:
        return (not self.is_off) and self.backend.startswith("landlab")

    @property
    def is_gospl(self) -> bool:
        return (not self.is_off) and self.backend.startswith("gospl")


@dataclass
class BridgeConfig:
    fluvial: SlotSpec = field(default_factory=lambda: SlotSpec("gospl.spl"))
    hillslope: SlotSpec = field(default_factory=lambda: SlotSpec("gospl.linear_diffusion"))
    mass_wasting: Optional[SlotSpec] = None
    """Optional Landlab ``BedrockLandslider`` (HyLands)."""

    flow: str = "gospl"
    """``gospl`` (map goSPL FA → Landlab) or ``landlab`` (PriorityFlood/FA)."""

    landlab_flow_component: str = "PriorityFloodFlowRouter"
    landlab_flow_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "flow_metric": "D8",
            "separate_hill_flow": True,
            "depression_handler": "fill",
            "accumulate_flow_hill": True,
        }
    )

    soil_policy: str = "auto"
    """``auto``: map soil if Landlab provides it; ``off``: ignore; ``require``: error if missing."""

    substeps: int = 1
    """Split goSPL ``dt`` into N Landlab substeps when explicit solvers need it."""

    gather_warn_nodes: int = 2_000_000
    """Warn if global node count exceeds this before rank-0 Landlab gather."""

    raster_scale: float = 1.0
    """Raster dx = mesh_dx / scale for landslide remap (API default 1.0)."""

    landlab_float32: bool = False
    """If True, irregular DDTD mirror uses float32 fields (Python components).

    Raster FA / BedrockLandslider stay float64 — Landlab Cython flow directors
    do not run correctly on float32 elev. goSPL/PETSc remain float64.
    """

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BridgeConfig":
        def _slot(key: str, default_backend: str) -> SlotSpec:
            raw = d.get(key, default_backend)
            # YAML may parse bare `off` as False
            if raw is None or raw is False or raw in ("off", "none", False):
                return SlotSpec("off")
            if isinstance(raw, str):
                if raw.lower() in {"off", "none", "disabled", "false"}:
                    return SlotSpec("off")
                return SlotSpec(raw)
            if isinstance(raw, dict):
                backend = raw.get("backend", default_backend)
                if backend is None or backend is False or backend in ("off", "none"):
                    return SlotSpec("off")
                return SlotSpec(str(backend), dict(raw.get("kwargs", {})))
            return SlotSpec(str(raw))

        mw = d.get("mass_wasting")
        mass = None
        if mw and mw not in ("off", "none"):
            if isinstance(mw, str):
                mass = SlotSpec(mw)
            else:
                mass = SlotSpec(mw["backend"], dict(mw.get("kwargs", {})))

        # landlab_float32 or legacy landlab_dtype: float32
        ll_f32 = bool(d.get("landlab_float32", False))
        if not ll_f32:
            dtype = str(d.get("landlab_dtype", "float64")).lower()
            ll_f32 = dtype in {"float32", "f4", "single", "f32"}

        return BridgeConfig(
            fluvial=_slot("fluvial", "gospl.spl"),
            hillslope=_slot("hillslope", "gospl.linear_diffusion"),
            mass_wasting=mass,
            flow=d.get("flow", "gospl"),
            landlab_flow_component=d.get("landlab_flow_component", "PriorityFloodFlowRouter"),
            landlab_flow_kwargs=dict(d.get("landlab_flow_kwargs", {})),
            soil_policy=d.get("soil_policy", "auto"),
            substeps=int(d.get("substeps", 1)),
            gather_warn_nodes=int(d.get("gather_warn_nodes", 2_000_000)),
            raster_scale=float(d.get("raster_scale", 1.0)),
            landlab_float32=ll_f32,
        )
