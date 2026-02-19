"""VTK algorithm source for lazy xarray DataArray evaluation.

Provides :class:`PyVistaXarraySource`, a VTK pipeline source that wraps
an xarray DataArray and lazily generates PyVista meshes on demand. This
enables level-of-detail rendering, time stepping, and spatial slicing
without loading the entire dataset into memory.
"""

from __future__ import annotations

import logging

import numpy as np
import pyvista as pv
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
import xarray as xr

logger = logging.getLogger(__name__)


class BaseSource(VTKPythonAlgorithmBase):
    """Base class for VTK Python algorithm sources with PyVista wrapping."""

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
        """Execute the algorithm and return the output mesh."""
        self.Update()
        return self.GetOutput()

    def update(self):
        """Alias for :meth:`Update`."""
        return self.Update()

    def get_output(self, port=0):
        """Alias for :meth:`GetOutput`."""
        return self.GetOutput(port=port)


class PyVistaXarraySource(BaseSource):
    """VTK algorithm source wrapping an xarray DataArray.

    Lazily evaluates the DataArray to produce a PyVista mesh on demand.
    Supports time stepping, resolution control, spatial slicing, and
    level-of-detail rendering for large or dask-backed datasets.

    Parameters
    ----------
    data_array : xr.DataArray, optional
        The xarray DataArray to visualize.
    x : str, optional
        Name of the X coordinate variable.
    y : str, optional
        Name of the Y coordinate variable.
    z : str, optional
        Name of the Z coordinate variable.
    time : str, optional
        Name of the time dimension for temporal indexing.
    order : str, default "C"
        Array memory layout for flattening (``"C"`` or ``"F"``).
    component : str, optional
        Name of an extra dimension to treat as vector components.
    mesh_type : str, optional
        Force a specific mesh type (``"rectilinear"``, ``"structured"``,
        or ``"points"``).
    resolution : float, optional
        Fraction of data points to include (0.0 to 1.0). Lower values
        produce coarser meshes for faster rendering.

    Examples
    --------
    >>> import xarray as xr
    >>> from pvxarray.vtk_source import PyVistaXarraySource
    >>> ds = xr.tutorial.load_dataset("air_temperature")
    >>> source = PyVistaXarraySource(ds.air, x="lon", y="lat", time="time")
    >>> mesh = source.apply()
    """

    def __init__(
        self,
        data_array: xr.DataArray | None = None,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        time: str | None = None,
        order: str = "C",
        component: str | None = None,
        mesh_type: str | None = None,
        resolution: float | None = None,
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
        self._mesh_type = mesh_type

        self._time = None
        self._time_index = 0
        if isinstance(time, str):
            self._time = time
        elif time is not None:
            raise TypeError("time must be a string or None")

        self._z_index = None
        self._slicing = None
        self._sliced_data_array = None
        self._persisted_data = None
        self._mesh = None

    def __str__(self):
        return (
            f"PyVistaXarraySource\n"
            f"  data_array: {self._data_array}\n"
            f"  resolution: {self._resolution}\n"
            f"  x: {self._x}\n"
            f"  y: {self._y}\n"
            f"  z: {self._z}\n"
            f"  order: {self._order}\n"
            f"  component: {self._component}\n"
            f"  time: {self._time}\n"
            f"  time_index: {self._time_index}"
        )

    @property
    def data_array(self):
        """The source xarray DataArray."""
        return self._data_array

    @data_array.setter
    def data_array(self, data_array):
        self._data_array = data_array
        self.Modified()

    @property
    def resolution(self):
        """Fraction of data points to include (0.0 to 1.0)."""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float):
        self._resolution = resolution
        self.Modified()

    @property
    def x(self):
        """Name of the X coordinate variable."""
        return self._x

    @x.setter
    def x(self, x: str):
        self._x = x
        self.Modified()

    @property
    def y(self):
        """Name of the Y coordinate variable."""
        return self._y

    @y.setter
    def y(self, y: str):
        self._y = y
        self.Modified()

    @property
    def z(self):
        """Name of the Z coordinate variable."""
        return self._z

    @z.setter
    def z(self, z: str):
        self._z = z
        self.Modified()

    @property
    def time(self):
        """Name of the time dimension."""
        return self._time

    @time.setter
    def time(self, time: str):
        self._time = time
        self.Modified()

    @property
    def order(self):
        """Array memory layout for flattening (``"C"`` or ``"F"``)."""
        return self._order

    @order.setter
    def order(self, order: str):
        self._order = order
        self.Modified()

    @property
    def component(self):
        """Name of the vector component dimension."""
        return self._component

    @component.setter
    def component(self, component: str):
        self._component = component
        self.Modified()

    @property
    def time_index(self):
        """Current time step index."""
        return self._time_index

    @time_index.setter
    def time_index(self, time_index: int):
        self._time_index = time_index
        self.Modified()

    @property
    def max_time_index(self):
        """Maximum valid time index, or ``None`` if no time dimension."""
        if self._time is not None:
            return len(self.data_array[self._time]) - 1
        return None

    @property
    def z_index(self):
        """Vertical level index for slicing, or ``None`` for all levels."""
        return self._z_index

    @z_index.setter
    def z_index(self, z_index: int):
        self._z_index = z_index
        self.Modified()

    @property
    def slicing(self):
        """Spatial slicing parameters as ``{dim: [start, stop, step]}``."""
        return self._slicing

    @slicing.setter
    def slicing(self, slicing: dict | None):
        self._slicing = slicing
        self.Modified()

    @property
    def sliced_data_array(self):
        """The DataArray after applying time, z-index, and spatial slicing."""
        if self._sliced_data_array is None:
            self._compute_sliced_data_array()
        return self._sliced_data_array

    @property
    def persisted_data(self):
        """The sliced data materialized into memory.

        For dask-backed arrays, calls ``.persist()`` to trigger
        computation. For in-memory arrays, returns the data as-is.
        """
        if self._persisted_data is None:
            da = self.sliced_data_array
            if da is not None and hasattr(da, "chunks") and da.chunks:
                self._persisted_data = da.persist()
            else:
                self._persisted_data = da
        return self._persisted_data

    @property
    def mesh(self):
        """The PyVista mesh generated from the persisted data."""
        if self._mesh is None:
            self._compute_mesh()
        return self._mesh

    @property
    def data_range(self):
        """``(min, max)`` of the persisted data values."""
        da = self.persisted_data
        return da.min(), da.max()

    def resolution_to_sampling_rate(self, data_array):
        """Convert a resolution fraction to per-axis sampling rates.

        Parameters
        ----------
        data_array : xr.DataArray
            The data array whose shape determines the sampling rate.

        Returns
        -------
        np.ndarray
            Array of 3 integers: sampling step for each axis,
            zero-padded if fewer than 3 dimensions.
        """
        shape = np.array(data_array.shape)
        n = np.floor(shape * self._resolution)
        rate = np.ceil(shape / n).astype(int)
        return np.pad(rate, (0, 3 - len(rate)), mode="constant")

    def _compute_sliced_data_array(self):
        if self.data_array is None:
            self._sliced_data_array = None
            return None

        indexing = {}
        if self._slicing is not None:
            indexing = {
                k: slice(*v) for k, v in self._slicing.items() if k in [self.x, self.y, self.z]
            }

        if self._time is not None:
            indexing[self._time] = self.time_index

        if self.z and self.z_index is not None:
            indexing[self.z] = self.z_index

        da = self.data_array.isel(indexing)

        if self._slicing is None and self._resolution is not None:
            rx, ry, rz = self.resolution_to_sampling_rate(da)
            if da.ndim <= 1:
                da = da[::rx]
            elif da.ndim == 2:
                da = da[::rx, ::ry]
            elif da.ndim == 3:
                da = da[::rx, ::ry, ::rz]

        self._sliced_data_array = da
        return self._sliced_data_array

    def _compute_mesh(self):
        self._mesh = self.persisted_data.pyvista.mesh(
            x=self._x,
            y=self._y,
            z=self._z if self._z_index is None else None,
            order=self._order,
            component=self._component,
            mesh_type=self._mesh_type,
            scales={k: v[2] for k, v in self._slicing.items()} if self._slicing else {},
        )
        return self._mesh

    def Modified(self, **kwargs):
        """Clear all cached data and mark the source as modified."""
        self._sliced_data_array = None
        self._persisted_data = None
        self._mesh = None
        super().Modified(**kwargs)

    def RequestData(self, request, inInfo, outInfo):
        """VTK pipeline callback to generate output data."""
        try:
            pdo = self.GetOutputData(outInfo, 0)
            pdo.ShallowCopy(self.mesh)
        except Exception:
            logger.exception("Error in PyVistaXarraySource.RequestData")
            raise
        return 1
