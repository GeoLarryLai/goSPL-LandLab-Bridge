"""Landlab dtype helpers (float32 mode for Python components; Cython FA stays float64)."""

from __future__ import annotations

from typing import Dict, Iterable, Type

import numpy as np

_PATCHED: set[int] = set()
_ORIGINAL_INFO: Dict[int, dict] = {}


def resolve_landlab_dtype(dtype) -> np.dtype:
    """Map config/name to numpy float32 or float64."""
    if isinstance(dtype, np.dtype):
        dt = dtype
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dt = np.dtype(dtype)
    else:
        key = str(dtype).lower().strip()
        if key in {"float32", "f4", "single", "f32"}:
            dt = np.dtype(np.float32)
        elif key in {"float64", "f8", "double", "float", "f64"}:
            dt = np.dtype(np.float64)
        else:
            raise ValueError(f"landlab_dtype must be float32 or float64, got {dtype!r}")
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"landlab_dtype must be float32 or float64, got {dt}")
    return dt


def patch_component_info_float32(cls: Type) -> None:
    """Rewrite Landlab component ``_info`` so ``float`` fields accept float32."""
    key = id(cls)
    if key in _PATCHED:
        return
    info = getattr(cls, "_info", None)
    if not info:
        _PATCHED.add(key)
        return
    _ORIGINAL_INFO[key] = {name: dict(meta) for name, meta in info.items()}
    new_info = {}
    for name, meta in info.items():
        m = dict(meta)
        if m.get("dtype") is float:
            m["dtype"] = np.float32
        new_info[name] = m
    cls._info = new_info
    _PATCHED.add(key)


def restore_component_info_dtype(cls: Type) -> None:
    """Undo float32 ``_info`` rewrite (needed before float64 raster DDTD in same kernel)."""
    key = id(cls)
    original = _ORIGINAL_INFO.get(key)
    if original is not None:
        cls._info = {name: dict(meta) for name, meta in original.items()}
    else:
        # Already stock, or patched before we tracked originals: coerce np.float32 → float
        info = getattr(cls, "_info", None)
        if info:
            new_info = {}
            for name, meta in info.items():
                m = dict(meta)
                if m.get("dtype") is np.float32:
                    m["dtype"] = float
                new_info[name] = m
            cls._info = new_info
    _PATCHED.discard(key)


def patch_components_float32(classes: Iterable[Type]) -> None:
    for cls in classes:
        patch_component_info_float32(cls)


def patch_ddtd_float32() -> None:
    """Allow DepthDependentTaylorDiffuser (Python) to use float32 node fields."""
    from landlab.components import DepthDependentTaylorDiffuser

    patch_component_info_float32(DepthDependentTaylorDiffuser)


def restore_ddtd_float64() -> None:
    """Restore DDTD ``_info`` for float64 grids (raster path / after float32 toggle)."""
    from landlab.components import DepthDependentTaylorDiffuser

    restore_component_info_dtype(DepthDependentTaylorDiffuser)
