"""Capability table and light probes for Landlab components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .loader import resolve_landlab_class


@dataclass
class Capability:
    name: str
    status: str  # supported | needs_landlab_flow | experimental | unsupported
    notes: str = ""


CAPABILITY_TABLE: Dict[str, Capability] = {
    "StreamPowerEroder": Capability(
        "StreamPowerEroder", "supported", "Works with goSPL-mapped SFD flow fields."
    ),
    "Space": Capability(
        "Space",
        "needs_landlab_flow",
        "Prefer flow: landlab; rejects route-to-multiple from goSPL MFD.",
    ),
    "SpaceLargeScaleEroder": Capability(
        "SpaceLargeScaleEroder",
        "needs_landlab_flow",
        "HyLands fluvial piece; use with PriorityFloodFlowRouter.",
    ),
    "BedrockLandslider": Capability(
        "BedrockLandslider",
        "needs_landlab_flow",
        "HyLands mass wasting; needs hill flow receivers (separate_hill_flow).",
    ),
    "LinearDiffuser": Capability(
        "LinearDiffuser", "supported", "Hillslope creep; no flow graph required."
    ),
    "DepthDependentTaylorDiffuser": Capability(
        "DepthDependentTaylorDiffuser",
        "supported",
        "Roering-style (Taylor) depth-dependent creep; needs soil__depth + soil_production__rate.",
    ),
    "TaylorNonLinearDiffuser": Capability(
        "TaylorNonLinearDiffuser",
        "supported",
        "Roering-style Taylor nonlinear creep on topography; linear_diffusivity ≈ D.",
    ),
    "ErosionDeposition": Capability(
        "ErosionDeposition",
        "needs_landlab_flow",
        "Rejects route-to-multiple; use SFD map or landlab flow.",
    ),
    "PriorityFloodFlowRouter": Capability(
        "PriorityFloodFlowRouter", "supported", "Opt-in Landlab flow backend."
    ),
    "FlowAccumulator": Capability(
        "FlowAccumulator", "supported", "Opt-in Landlab flow backend."
    ),
}


def probe_component(name: str) -> Capability:
    """Resolve import and return table entry (or experimental if unknown)."""
    short = name.split(".")[-1]
    try:
        resolve_landlab_class(short)
    except Exception as e:
        return Capability(short, "unsupported", f"import failed: {e}")
    if short in CAPABILITY_TABLE:
        return CAPABILITY_TABLE[short]
    return Capability(short, "experimental", "Not in seed table; try via probe.")


def list_supported() -> List[str]:
    return [k for k, v in CAPABILITY_TABLE.items() if v.status != "unsupported"]
