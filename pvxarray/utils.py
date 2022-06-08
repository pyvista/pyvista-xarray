import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


def soa(*arrays):
    """Combine separately allocated numpy arrays into a VTK SOA array."""
    if len(arrays) == 0:
        raise ValueError("No arrays passed")
    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"argument {i} is not a numpy array")
    if len({arr.dtype for arr in arrays}) != 1:
        raise TypeError("All components must have the same dtype")
    if len({arr.size for arr in arrays}) != 1:
        raise TypeError("All components must have the same size")
    if len({arr.ndim for arr in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("Please ravel the input arrays")
    data = vtk.vtkSOADataArrayTemplate[arrays[0].dtype]()
    data.SetNumberOfComponents(len(arrays))
    data.SetNumberOfValues(arrays[0].size)
    for i, arr in enumerate(arrays):
        data.SetArray(i, numpy_to_vtk(arr), arr.size, True, True)
    return data


def soa_points(x, y, z):
    """Generate vtkPoints from 3 separately allocated numpy arrays."""
    data = soa(x, y, z)
    points = vtk.vtkPoints()
    points.SetData(data)
    return points
