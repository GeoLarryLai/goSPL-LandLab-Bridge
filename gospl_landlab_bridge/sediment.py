"""Sediment / soil writeback (policy B+)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def elevation_delta_to_volume(dz: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Positive volume = deposition; negative = erosion (m³ over the step)."""
    return np.asarray(dz, dtype=float) * np.asarray(area, dtype=float)


def extract_landlab_sediment(grid) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (outflux, soil_thickness) if present on the Landlab grid.

    outflux preferred keys: sediment__outflux, sediment__flux
    soil: soil__depth, soil_thickness
    """
    outflux = None
    soil = None
    for key in ("sediment__outflux", "sediment__flux", "sediment__outflux_total"):
        if key in grid.at_node:
            outflux = np.asarray(grid.at_node[key], dtype=float).copy()
            break
    for key in ("soil__depth", "soil_thickness", "soil__thickness"):
        if key in grid.at_node:
            soil = np.asarray(grid.at_node[key], dtype=float).copy()
            break
    return outflux, soil


def choose_sediment_volume(
    *,
    dz: np.ndarray,
    area: np.ndarray,
    landlab_outflux: Optional[np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    B+: prefer Landlab outflux (assumed m³/yr) * dt; else -dz*area for eroded mass
    injected to cascade (erosion → positive sediment source).
    """
    if landlab_outflux is not None:
        return np.asarray(landlab_outflux, dtype=float) * float(dt)
    # eroded volume becomes sediment; deposited volume is negative source
    return -elevation_delta_to_volume(dz, area)
