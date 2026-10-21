"""Resolve and instantiate Landlab component classes by name."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Type


def resolve_landlab_class(name: str) -> Type:
    """
    Resolve ``Space``, ``landlab.components.Space``, or ``BedrockLandslider``.
    """
    name = name.strip()
    if name.startswith("landlab."):
        # landlab.Space → treat as component short name if only one more part
        parts = name.split(".")
        if len(parts) == 2:
            name = parts[1]
        else:
            mod = importlib.import_module(".".join(parts[:-1]))
            return getattr(mod, parts[-1])

    # Try landlab.components namespace (public exports)
    try:
        import landlab.components as comps

        if hasattr(comps, name):
            return getattr(comps, name)
    except Exception:
        pass

    # Explicit known map (HyLands + common erosers)
    known = {
        "Space": "landlab.components.space.space:Space",
        "SpaceLargeScaleEroder": "landlab.components.space.space_large_scale_eroder:SpaceLargeScaleEroder",
        "BedrockLandslider": "landlab.components.bedrock_landslider.bedrock_landslider:BedrockLandslider",
        "StreamPowerEroder": "landlab.components.stream_power.stream_power:StreamPowerEroder",
        "LinearDiffuser": "landlab.components.diffusion.diffusion:LinearDiffuser",
        "DepthDependentTaylorDiffuser": "landlab.components.depth_dependent_taylor_soil_creep.hillslope_depth_dependent_taylor_flux:DepthDependentTaylorDiffuser",
        "TaylorNonLinearDiffuser": "landlab.components.taylor_nonlinear_hillslope_flux.taylor_nonlinear_hillslope_flux:TaylorNonLinearDiffuser",
        "ErosionDeposition": "landlab.components.erosion_deposition.erosion_deposition:ErosionDeposition",
        "PriorityFloodFlowRouter": "landlab.components.priority_flood_flow_router.priority_flood_flow_router:PriorityFloodFlowRouter",
        "FlowAccumulator": "landlab.components.flow_accum.flow_accumulator:FlowAccumulator",
    }
    if name in known:
        modname, attr = known[name].split(":")
        mod = importlib.import_module(modname)
        return getattr(mod, attr)

    # Common nested modules
    candidates = [
        f"landlab.components.{_snake(name)}",
        f"landlab.components.{name.lower()}",
    ]
    last_err = None
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, name):
                return getattr(mod, name)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(f"Could not resolve Landlab component {name!r} ({last_err})")


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _coerce_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            try:
                if v.lower() in ("true", "false"):
                    out[k] = v.lower() == "true"
                elif "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
                continue
            except Exception:
                pass
        out[k] = v
    return out


def instantiate_component(grid, class_name: str, kwargs: Dict[str, Any]):
    cls = resolve_landlab_class(class_name)
    return cls(grid, **_coerce_kwargs(dict(kwargs)))
