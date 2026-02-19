"""Xarray DataArray accessor for PyVista 3D visualization.

Registers the ``.pyvista`` namespace on :class:`xarray.DataArray` objects,
providing methods to convert xarray data into PyVista mesh objects for
interactive 3D rendering.

Example
-------
>>> import pvxarray  # noqa: F401
>>> import xarray as xr
>>> ds = xr.tutorial.load_dataset("air_temperature")
>>> da = ds.air[{"time": 0}]
>>> mesh = da.pyvista.mesh(x="lon", y="lat")
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import xarray as xr

from pvxarray import points, rectilinear, structured
from pvxarray.cf import detect_axes
from pvxarray.vtk_source import PyVistaXarraySource

methods = {
    "points": points.mesh,
    "rectilinear": rectilinear.mesh,
    "structured": structured.mesh,
}


class _LocIndexer:
    """Location-based indexer delegating to the underlying DataArray."""

    def __init__(self, parent: PyVistaAccessor):
        self.parent = parent

    def __getitem__(self, key) -> xr.DataArray:
        return self.parent._obj.loc[key]

    def __setitem__(self, key, value) -> None:
        self.parent._obj.loc[key] = value


@xr.register_dataarray_accessor("pyvista")
class PyVistaAccessor:
    """PyVista accessor for :class:`xarray.DataArray`.

    Adds a ``.pyvista`` namespace to DataArray objects with methods for
    creating 3D meshes, plotting, and constructing VTK algorithm sources.

    Parameters
    ----------
    xarray_obj : xr.DataArray
        The DataArray this accessor is attached to.

    Notes
    -----
    Import ``pvxarray`` to register this accessor::

        import pvxarray  # noqa: F401
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """Initialize the accessor with the parent DataArray."""
        self._obj = xarray_obj
        self._mesh = None

    def __getitem__(self, key):
        """Index into the underlying DataArray."""
        return self._obj.__getitem__(key)

    @property
    def loc(self) -> _LocIndexer:
        """Location-based indexer, mirroring :attr:`xarray.DataArray.loc`."""
        return _LocIndexer(self)

    @property
    def data(self) -> np.ndarray:
        """The underlying NumPy array of the DataArray values.

        Ensures native byte order for VTK compatibility. Remote data
        sources (e.g. OPeNDAP) may return big-endian arrays that VTK
        cannot interpret correctly.
        """
        values = self._obj.values
        if np.issubdtype(values.dtype, np.number) and not values.dtype.isnative:
            values = values.astype(values.dtype.newbyteorder("="))
        return values

    @property
    def axes(self) -> dict[str, str]:
        """Detected CF axis mapping for this DataArray's coordinates.

        Returns a dictionary mapping axis types (``"X"``, ``"Y"``,
        ``"Z"``, ``"T"``) to coordinate variable names. Detection uses
        CF attributes (``axis``, ``standard_name``, ``units``) and
        falls back to variable name heuristics.

        Returns
        -------
        dict[str, str]
            e.g. ``{"X": "lon", "Y": "lat", "T": "time"}``
        """
        return detect_axes(self._obj)

    @property
    def spatial_coords(self) -> list[str]:
        """Names of detected spatial coordinate variables (X, Y, Z)."""
        return [v for k, v in self.axes.items() if k in ("X", "Y", "Z")]

    def _get_array(self, key: str, scale: float = 1) -> np.ndarray:
        """Retrieve coordinate values as a NumPy array.

        For non-numeric coordinates (e.g. datetime, string), returns
        an array of scaled integer indices instead.

        Parameters
        ----------
        key : str
            Name of the coordinate variable.
        scale : float, default 1
            Scale factor applied to index-based arrays for non-numeric
            coordinates.

        Returns
        -------
        np.ndarray
            The coordinate values or scaled indices.

        Raises
        ------
        KeyError
            If *key* is not a coordinate on this DataArray.
        """
        try:
            values = self._obj[key].values
            if not np.issubdtype(values.dtype, np.number):
                # Non-numeric coordinate: use scaled indices
                values = np.arange(len(values), dtype=float) * scale
            elif not values.dtype.isnative:
                # VTK requires native byte order; remote data (e.g.
                # OPeNDAP/THREDDS) often arrives big-endian which VTK
                # would misinterpret as garbage values.
                values = values.astype(values.dtype.newbyteorder("="))
            return values
        except KeyError as e:
            raise KeyError(
                f"Key {key!r} not present in DataArray. "
                f"Available coordinates: {list(self._obj.coords.keys())}"
            ) from e

    def mesh(
        self,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        order: str | None = None,
        component: str | None = None,
        mesh_type: str | None = None,
        scales: dict | None = None,
    ) -> pv.DataSet:
        """Create a PyVista mesh from this DataArray.

        Converts the DataArray into a :class:`pyvista.RectilinearGrid`,
        :class:`pyvista.StructuredGrid`, or :class:`pyvista.PolyData`
        depending on the coordinate dimensionality and *mesh_type*.

        When *x*, *y*, and *z* are all ``None``, coordinates are
        auto-detected from CF-convention attributes and variable names.

        Parameters
        ----------
        x : str, optional
            Name of the coordinate to use as the X axis.
        y : str, optional
            Name of the coordinate to use as the Y axis.
        z : str, optional
            Name of the coordinate to use as the Z axis.
        order : str, optional
            Array memory layout for flattening (``"C"`` or ``"F"``).
        component : str, optional
            Name of an extra dimension to treat as vector components
            (e.g. ``"band"`` for RGB data).
        mesh_type : str, optional
            Force a specific mesh type: ``"rectilinear"``,
            ``"structured"``, or ``"points"``. Auto-detected from
            coordinate dimensionality when ``None``.
        scales : dict, optional
            Mapping of coordinate names to scale factors, applied
            when coordinates are non-numeric (replaced by indices).

        Returns
        -------
        pyvista.DataSet
            A PyVista mesh object containing the data and coordinates.

        Raises
        ------
        KeyError
            If a specified coordinate name is not found, or
            *mesh_type* is not a recognized type.
        ValueError
            If no spatial coordinates can be determined.
        """
        # Auto-detect coordinates if none specified
        if x is None and y is None and z is None:
            axes = detect_axes(self._obj)
            x = axes.get("X")
            y = axes.get("Y")
            z = axes.get("Z")
            if x is None and y is None:
                raise ValueError(
                    "Could not auto-detect spatial coordinates. "
                    "Specify x=, y=, and/or z= explicitly. "
                    f"Available coordinates: {list(self._obj.coords.keys())}"
                )

        ndim = 0
        if x is not None:
            _x = self._get_array(x)
            ndim = _x.ndim
        if y is not None:
            _y = self._get_array(y)
            if _y.ndim > ndim:
                ndim = _y.ndim
        if z is not None:
            _z = self._get_array(z)
            if _z.ndim > ndim:
                ndim = _z.ndim
        if mesh_type is None:  # Auto-detect mesh type from coord dimensionality
            mesh_type = "structured" if ndim > 1 else "rectilinear"
        try:
            meth = methods[mesh_type]
        except KeyError as e:
            raise KeyError(
                f"Unknown mesh_type {mesh_type!r}. Choose from: {list(methods.keys())}"
            ) from e
        return meth(self, x=x, y=y, z=z, order=order, component=component, scales=scales)

    def plot(
        self,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        order: str = "C",
        component: str | None = None,
        mesh_type: str | None = None,
        **kwargs,
    ):
        """Create a mesh and immediately plot it.

        Convenience method that calls :meth:`mesh` and then
        :meth:`pyvista.DataSet.plot`. All extra keyword arguments
        are forwarded to PyVista's ``plot()`` method.

        Parameters
        ----------
        x, y, z : str, optional
            Coordinate names (see :meth:`mesh`).
        order : str, default "C"
            Array memory layout for flattening.
        component : str, optional
            Vector component dimension name.
        mesh_type : str, optional
            Force a specific mesh type.
        **kwargs
            Passed to :meth:`pyvista.DataSet.plot`.
        """
        return self.mesh(x=x, y=y, z=z, order=order, component=component, mesh_type=mesh_type).plot(
            **kwargs
        )

    def algorithm(
        self,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        time: str | None = None,
        order: str = "C",
        component: str | None = None,
        mesh_type: str | None = None,
        resolution: float = 1.0,
    ) -> PyVistaXarraySource:
        """Create a VTK algorithm source for lazy mesh generation.

        Returns a :class:`~pvxarray.PyVistaXarraySource` that lazily
        evaluates data on demand, supporting time stepping, resolution
        control, and spatial slicing. This is ideal for large or
        dask-backed datasets.

        Parameters
        ----------
        x, y, z : str, optional
            Coordinate names for spatial axes.
        time : str, optional
            Name of the time dimension for temporal slicing.
        order : str, default "C"
            Array memory layout for flattening.
        component : str, optional
            Vector component dimension name.
        mesh_type : str, optional
            Force a specific mesh type.
        resolution : float, default 1.0
            Fraction of data points to include (0.0 to 1.0).
            Lower values produce coarser meshes for faster rendering.

        Returns
        -------
        PyVistaXarraySource
            A VTK algorithm source that can be added to a PyVista
            plotter or evaluated with ``.apply()``.
        """
        return PyVistaXarraySource(
            data_array=self._obj,
            x=x,
            y=y,
            z=z,
            time=time,
            order=order,
            component=component,
            mesh_type=mesh_type,
            resolution=resolution,
        )
