"""Install a lightweight vtk stub before importing goSPL when real vtk SIGKILLs."""

from __future__ import annotations

import sys
import types

import numpy as np


class _FakeArray:
    def __init__(self, data=None, deep=False, array_type=None):
        self.data = np.asarray([] if data is None else data)

    def GetDataType(self):
        return 0


class _FakePoints:
    def SetData(self, arr):
        self._arr = arr


class _FakeCellArray:
    def SetCells(self, n, connectivity):
        self._n = n
        self._c = connectivity


class _FakeUnstructuredGrid:
    def SetPoints(self, pts):
        self._pts = pts

    def SetCells(self, *args):
        self._cells = args


class _FakeVtkObject:
    @staticmethod
    def GlobalWarningDisplayOff():
        return None


class _FakeContourFilter:
    def SetInputData(self, *args, **kwargs):
        return None

    def SetValue(self, *args, **kwargs):
        return None

    def Update(self):
        return None

    def GetOutput(self):
        return None


class _FakeVtkModule(types.ModuleType):
    __gospl_landlab_shim__ = True
    VTK_TRIANGLE = 5

    class vtkVersion:
        @staticmethod
        def GetVTKVersion() -> str:
            return "gospl_landlab-shim"

    vtkObject = _FakeVtkObject
    vtkContourFilter = _FakeContourFilter
    vtkUnstructuredGrid = _FakeUnstructuredGrid
    vtkPoints = _FakePoints
    vtkCellArray = _FakeCellArray
    vtkUnsignedCharArray = _FakeArray
    vtkIdTypeArray = _FakeArray


def install_vtk_shim(force: bool = False) -> bool:
    """
    Put stub modules in ``sys.modules`` for ``vtk`` and ``vtk.util.numpy_support``.

    Provides enough surface for goSPL ``_generateVTKmesh`` (clinSlp path).
    Call **before** ``from gospl.model import Model``.
    """
    if (not force) and "vtk" in sys.modules:
        mod = sys.modules["vtk"]
        if getattr(mod, "__gospl_landlab_shim__", False):
            return True
        return False

    vtk_mod = _FakeVtkModule("vtk")
    sys.modules["vtk"] = vtk_mod

    util = types.ModuleType("vtk.util")
    util.__gospl_landlab_shim__ = True  # type: ignore[attr-defined]
    sys.modules["vtk.util"] = util

    ns = types.ModuleType("vtk.util.numpy_support")
    ns.__gospl_landlab_shim__ = True  # type: ignore[attr-defined]

    def numpy_to_vtk(array, deep=True, array_type=None):
        return _FakeArray(array, deep=deep, array_type=array_type)

    def numpy_to_vtkIdTypeArray(array, deep=1):
        return _FakeArray(array, deep=deep)

    def vtk_to_numpy(obj):
        return np.asarray(getattr(obj, "data", []))

    ns.numpy_to_vtk = numpy_to_vtk  # type: ignore[attr-defined]
    ns.numpy_to_vtkIdTypeArray = numpy_to_vtkIdTypeArray  # type: ignore[attr-defined]
    ns.vtk_to_numpy = vtk_to_numpy  # type: ignore[attr-defined]
    sys.modules["vtk.util.numpy_support"] = ns
    return True
