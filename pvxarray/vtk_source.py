import traceback
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
        data_array: Optional[xr.DataArray] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        time: Optional[str] = None,
        order: str = "C",
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

        self._z_index = None

    def __str__(self):
        return f"""
data_array: {self._data_array}
resolution: {self._resolution}
x: {self._x}
y: {self._y}
z: {self._z}
order: {self._order}
component: {self._component}
time: {self._time}
time_index: {self._time_index}
"""

    @property
    def data_array(self):
        return self._data_array

    @data_array.setter
    def data_array(self, data_array):
        self._data_array = data_array
        self.Modified()

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: int):
        self._resolution = resolution
        self.Modified()

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x: str):
        self._x = x
        self.Modified()

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y: str):
        self._y = y
        self.Modified()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z: str):
        self._z = z
        self.Modified()

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time: str):
        self._time = time
        self.Modified()

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order: str):
        self._order = order
        self.Modified()

    @property
    def component(self):
        return self._component

    @component.setter
    def component(self, component: str):
        self._component = component
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

    @property
    def z_index(self):
        return self._z_index

    @z_index.setter
    def z_index(self, z_index: int):
        self._z_index = z_index
        self.Modified()

    @property
    def data_range(self):
        return self.data_array.min(), self.data_array.max()

    def RequestData(self, request, inInfo, outInfo):
        # Use open data_array handle to fetch data at
        # desired Level of Detail
        try:
            if self._time is not None:
                da = self.data_array[{self._time: self.time_index}]
            else:
                da = self.data_array

            if self._z and self._z_index is not None:
                da = da[{self._z: self.z_index}]

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
                x=self._x,
                y=self._y,
                z=self._z if self._z_index is None else None,
                order=self._order,
                component=self._component,
            )

            pdo = self.GetOutputData(outInfo, 0)
            pdo.ShallowCopy(mesh)
        except Exception as e:
            traceback.print_exc()
            raise e
        return 1
