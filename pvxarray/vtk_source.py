"""VTK algorithm source for lazy xarray DataArray evaluation.

Provides :class:`PyVistaXarraySource`, a VTK pipeline source that wraps
an xarray DataArray and lazily generates PyVista meshes on demand. This
enables level-of-detail rendering, time stepping, and spatial slicing
without loading the entire dataset into memory.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pyvista as pv
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkFiltersCore import vtkArrayCalculator
import xarray as xr

from pvxarray.cf import is_bounds_variable

logger = logging.getLogger(__name__)


def _format_time_labels(time_coord: xr.DataArray) -> list[str]:
    """Format time coordinate values as human-readable strings.

    Parameters
    ----------
    time_coord : xr.DataArray
        The time coordinate array.

    Returns
    -------
    list[str]
        Formatted time labels. For datetime-like coordinates,
        uses "YYYY-MM-DD HH:MM:SS" format. For numeric or other
        coordinates, uses string representation.
    """
    values = time_coord.values
    if np.issubdtype(values.dtype, np.datetime64):
        return [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in values]
    return [str(v) for v in values]


class BaseSource(VTKPythonAlgorithmBase):
    """Base class for VTK Python algorithm sources with PyVista wrapping."""

    def __init__(self, nOutputPorts=1, outputType="vtkTable", **kwargs):
        """Initialize the base source with no input ports."""
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=nOutputPorts, outputType=outputType, **kwargs
        )

    def GetOutput(self, port=0):
        """Return the output wrapped as a PyVista mesh."""
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
    dataset : xr.Dataset, optional
        Parent Dataset for multi-array support. When provided along
        with *arrays*, additional data variables are loaded onto the
        mesh alongside the primary *data_array*.
    arrays : list[str], optional
        Names of extra data variables from *dataset* to load onto
        the mesh.

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
        dataset: xr.Dataset | None = None,
        arrays: list[str] | None = None,
    ):
        """Initialize the source with a DataArray and coordinate mapping."""
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

        self._dataset = dataset
        self._arrays = list(arrays) if arrays else []
        self._computed: dict[str, str] = {}
        self._pipeline: list = []

    def __str__(self):
        """Return a human-readable summary of the source configuration."""
        parts = [
            "PyVistaXarraySource",
            f"  data_array: {self._data_array}",
            f"  resolution: {self._resolution}",
            f"  x: {self._x}",
            f"  y: {self._y}",
            f"  z: {self._z}",
            f"  order: {self._order}",
            f"  component: {self._component}",
            f"  time: {self._time}",
            f"  time_index: {self._time_index}",
        ]
        if self._arrays:
            parts.append(f"  arrays: {self._arrays}")
        if self._computed:
            parts.append(f"  computed: {[k for k in self._computed if not k.startswith('_')]}")
        if self._pipeline:
            parts.append(f"  pipeline: {len(self._pipeline)} filter(s)")
        return "\n".join(parts)

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
    def time_labels(self) -> list[str] | None:
        """Human-readable labels for all time steps.

        Returns ``None`` if no time dimension is set.
        """
        if self._time is None or self._data_array is None:
            return None
        return _format_time_labels(self._data_array[self._time])

    @property
    def time_label(self) -> str | None:
        """Human-readable label for the current time step.

        Returns ``None`` if no time dimension is set.
        """
        labels = self.time_labels
        if labels is None:
            return None
        return labels[self._time_index]

    @property
    def dataset(self) -> xr.Dataset | None:
        """Parent :class:`xr.Dataset` for multi-array support."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: xr.Dataset | None):
        self._dataset = dataset
        self.Modified()

    @property
    def arrays(self) -> list[str]:
        """Names of extra data variables to load onto the mesh."""
        return self._arrays

    @arrays.setter
    def arrays(self, arrays: list[str]):
        self._arrays = list(arrays) if arrays else []
        self.Modified()

    @property
    def available_arrays(self) -> list[str]:
        """Data variables from *dataset* with the same dimensions as *data_array*.

        Excludes CF boundary variables (names ending in ``_bnds``,
        ``_bounds``, ``_vertices``).

        Returns an empty list if *dataset* or *data_array* is not set.
        """
        if self._dataset is None:
            return []
        if self._data_array is None:
            return [name for name in self._dataset.data_vars if not is_bounds_variable(name)]
        target_dims = set(self._data_array.dims)
        return [
            name
            for name in self._dataset.data_vars
            if set(self._dataset[name].dims) == target_dims and not is_bounds_variable(name)
        ]

    @property
    def computed(self) -> dict[str, str]:
        """Computed fields as ``{name: expression}`` pairs.

        Expressions use ``vtkArrayCalculator`` syntax. Input array
        names from :attr:`arrays` or the primary data array can be
        referenced directly in expressions.

        A special key ``"_use_scalars"`` maps to a list of array
        names the calculator should recognize as scalar variables.

        Examples
        --------
        >>> source.arrays = ["u", "v"]
        >>> source.computed = {
        ...     "_use_scalars": ["u", "v"],
        ...     "speed": "sqrt(u*u + v*v)",
        ... }
        """
        return self._computed

    @computed.setter
    def computed(self, computed: dict[str, str]):
        self._computed = computed or {}
        self.Modified()

    @property
    def pipeline(self) -> list:
        """List of VTK filters or callables to apply after mesh creation.

        Each element is either:

        - A VTK algorithm object with ``SetInputData()``,
          ``Update()``, and ``GetOutput()`` methods.
        - A callable ``(pv.DataSet) -> pv.DataSet``.

        Filters are applied in order after computed fields.

        Examples
        --------
        >>> source.pipeline = [lambda mesh: mesh.warp_by_scalar(factor=0.001)]
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: list):
        self._pipeline = list(pipeline)
        self.Modified()

    @property
    def state(self) -> dict:
        """Export the source configuration as a JSON-serializable dict.

        Captures all configuration properties except the underlying
        data and pipeline filters. Use :meth:`load_state` to restore.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "x": self._x,
            "y": self._y,
            "z": self._z,
            "time": self._time,
            "time_index": self._time_index,
            "z_index": self._z_index,
            "order": self._order,
            "component": self._component,
            "mesh_type": self._mesh_type,
            "resolution": self._resolution,
            "slicing": self._slicing,
            "arrays": self._arrays,
            "computed": self._computed,
        }

    def load_state(self, state: dict) -> None:
        """Restore source configuration from a state dict.

        Parameters
        ----------
        state : dict
            Configuration dictionary as returned by :attr:`state`.
            Unknown keys are silently ignored.
        """
        simple_keys = (
            "x",
            "y",
            "z",
            "time",
            "order",
            "component",
            "mesh_type",
            "resolution",
            "slicing",
            "arrays",
            "computed",
        )
        for key in simple_keys:
            if key in state:
                setattr(self, f"_{key}", state[key])
        if "time_index" in state:
            self._time_index = state["time_index"]
        if "z_index" in state:
            self._z_index = state["z_index"]
        self.Modified()

    def to_json(self) -> str:
        """Serialize state to a JSON string."""
        return json.dumps(self.state, indent=2)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        data_array: xr.DataArray | None = None,
        dataset: xr.Dataset | None = None,
    ) -> PyVistaXarraySource:
        """Create a source from a JSON state string.

        Parameters
        ----------
        json_str : str
            JSON string as produced by :meth:`to_json`.
        data_array : xr.DataArray, optional
            The data array for the source.
        dataset : xr.Dataset, optional
            The dataset for multi-array support.

        Returns
        -------
        PyVistaXarraySource
        """
        state = json.loads(json_str)
        source = cls(data_array=data_array, dataset=dataset)
        source.load_state(state)
        return source

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

    def _build_indexing(self) -> dict:
        """Build the isel indexing dict from current slicing/time/z state."""
        indexing = {}
        if self._slicing is not None:
            indexing = {
                k: slice(*v) for k, v in self._slicing.items() if k in [self.x, self.y, self.z]
            }
        if self._time is not None:
            indexing[self._time] = self.time_index
        if self.z and self.z_index is not None:
            indexing[self.z] = self.z_index
        return indexing

    def _subsample(self, da: xr.DataArray) -> xr.DataArray:
        """Apply resolution-based subsampling to a DataArray."""
        if self._slicing is not None or self._resolution is None:
            return da
        rx, ry, rz = self.resolution_to_sampling_rate(da)
        if da.ndim <= 1:
            return da[::rx]
        elif da.ndim == 2:
            return da[::rx, ::ry]
        elif da.ndim == 3:
            return da[::rx, ::ry, ::rz]
        return da

    def _compute_sliced_data_array(self):
        if self.data_array is None:
            self._sliced_data_array = None
            return None

        indexing = self._build_indexing()
        da = self.data_array.isel(indexing)
        da = self._subsample(da)

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

        if self._dataset is not None and self._arrays:
            primary_name = self._data_array.name or "data"
            indexing = self._build_indexing()
            for array_name in self._arrays:
                if array_name == primary_name or array_name not in self._dataset:
                    continue
                da = self._subsample(self._dataset[array_name].isel(indexing))
                values = da.values
                if np.issubdtype(values.dtype, np.number) and not values.dtype.isnative:
                    values = values.astype(values.dtype.newbyteorder("="))
                self._mesh[array_name] = values.ravel(order=self._order)

        self._mesh = self._apply_computed_fields(self._mesh)
        self._mesh = self._apply_pipeline(self._mesh)
        return self._mesh

    def _apply_computed_fields(self, mesh: pv.DataSet) -> pv.DataSet:
        """Apply vtkArrayCalculator for each computed expression."""
        if not self._computed:
            return mesh

        use_scalars = self._computed.get("_use_scalars", [])
        for name, expression in self._computed.items():
            if name.startswith("_"):
                continue
            calc = vtkArrayCalculator()
            calc.SetInputData(mesh)
            calc.SetAttributeTypeToPointData()
            for arr_name in mesh.point_data:
                if not use_scalars or arr_name in use_scalars:
                    calc.AddScalarVariable(arr_name, arr_name, 0)
            calc.SetFunction(expression)
            calc.SetResultArrayName(name)
            calc.Update()
            mesh = pv.wrap(calc.GetOutput())
        return mesh

    def _apply_pipeline(self, mesh: pv.DataSet) -> pv.DataSet:
        """Apply pipeline filters sequentially to the mesh."""
        for filt in self._pipeline:
            if callable(filt) and not hasattr(filt, "SetInputData"):
                mesh = filt(mesh)
                if not isinstance(mesh, pv.DataSet):
                    mesh = pv.wrap(mesh)
            else:
                filt.SetInputData(mesh)
                filt.Update()
                mesh = pv.wrap(filt.GetOutput())
        return mesh

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
