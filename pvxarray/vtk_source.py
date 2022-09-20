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
        time: Optional[str] = None,
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

        self._time = None
        self._time_index = 0
        if isinstance(time, str):
            self._time = time
        elif time is not None:
            raise TypeError

    @property
    def data_array(self):
        return self._data_array

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int):
        self._resolution = resolution
        self.Modified()

    def resolution_to_sampling_rate(self, data_array):
        """Convert percentage to sampling rate."""
        shape = np.array(data_array.shape)
        n = np.floor(shape * self._resolution)
        rate = np.ceil(shape / n).astype(int)
        return np.pad(rate, (0, 3 - len(rate)), mode="constant")

    @property
    def time_index(self):
        return self._time_index

    @time_index.setter
    def time_index(self, time_index: int):
        # TODO: hook into the VTK pipeling to get requested time
        self._time_index = time_index
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        # Use open data_array handle to fetch data at
        # desired Level of Detail
        if self._time is not None:
            da = self.data_array[{self._time: self.time_index}]
        else:
            da = self.data_array

        rx, ry, rz = self.resolution_to_sampling_rate(da)
        if da.ndim == 1:
            da = da[::rx]
        elif da.ndim == 2:
            da = da[::rx, ::ry]
        elif da.ndim == 3:
            da = da[::rx, ::ry, ::rz]
        else:
            raise ValueError

        mesh = da.pyvista.mesh(
            x=self._x, y=self._y, z=self._z, order=self._order, component=self._component
        )

        pdo = self.GetOutputData(outInfo, 0)
        pdo.ShallowCopy(mesh)
        return 1
