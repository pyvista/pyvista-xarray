from typing import Optional

import numpy as np
import pyvista as pv
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase
import xarray as xr


class BaseSource(VTKPythonAlgorithmBase):
    def __init__(self, nOutputPorts=1, outputType="vtkTable", **kwargs):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=nOutputPorts, outputType=outputType, **kwargs
        )

    def GetOutput(self, port=0):
        output = pv.wrap(self.GetOutputDataObject(port))
        if output.active_scalars is None and output.n_arrays:
            if len(output.point_data):
                output.set_active_scalars(output.point_data.keys()[0])
            elif len(output.cell_data):
                output.set_active_scalars(output.cell_data.keys()[0])
        return output

    def apply(self):
        self.Update()
        return self.GetOutput()

    def update(self):
        """Alias for self.Update()"""
        return self.Update()

    def get_output(self, port=0):
        """Alias for self.GetOutput()"""
        return self.GetOutput(port=port)


class PyVistaXarraySource(BaseSource):
    def __init__(
        self,
        data_array: xr.DataArray,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        order: Optional[str] = "C",
        component: Optional[str] = None,
        resolution: float = 1.0,
        **kwargs,
    ):
        BaseSource.__init__(
            self,
            nOutputPorts=1,
            outputType="vtkRectilinearGrid",
        )
        self._data_array = data_array
        self._resolution = resolution

        self._x = x
        self._y = y
        self._z = z
        self._order = order
        self._component = component

    @property
    def dataset(self):
        return self._dataset

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int):
        self._resolution = resolution
        self.Modified()

    def resolution_to_sampling_rate(self):
        shape = np.array(self._data_array.shape)
        n = np.floor(shape * self._resolution).astype(int)
        rate = shape // n
        return np.pad(rate, (0, 3 - len(rate)), mode="constant")

    def RequestData(self, request, inInfo, outInfo):
        # Use open data_array handle to fetch data at desired LOD
        rx, ry, rz = self.resolution_to_sampling_rate()
        if self._data_array.ndim == 1:
            da = self._data_array[::rx]
        elif self._data_array.ndim == 2:
            da = self._data_array[::rx, ::ry]
        elif self._data_array.ndim == 3:
            da = self._data_array[::rx, ::ry, ::rz]
        else:
            raise ValueError

        mesh = da.pyvista.mesh(
            x=self._x, y=self._y, z=self._z, order=self._order, component=self._component
        )

        pdo = self.GetOutputData(outInfo, 0)
        pdo.ShallowCopy(mesh)
        return 1
