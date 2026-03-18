"""Create PyVista StructuredGrid meshes from xarray DataArrays.

StructuredGrid handles curvilinear coordinates — where coordinate
arrays are 2D or 3D (e.g. ``lon_rho(xi, eta)``). The grid has
logical i/j/k structure but physical points can be arbitrarily
positioned in space.

.. warning::
    StructuredGrid creation always copies data because VTK stores
    points as an interleaved ``(N, 3)`` array, which requires
    rearranging the source coordinate arrays.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyvista as pv

from pvxarray.errors import DataCopyWarning


def _coerce_shapes(*arrs):
    """Broadcast coordinate arrays to the same shape.

    When mixing 1D and 2D/3D coordinates (e.g. a 1D depth array
    with 2D lat/lon), this repeats the lower-dimensional arrays
    to match the highest-dimensional one.

    Parameters
    ----------
    *arrs : np.ndarray or None
        Coordinate arrays. ``None`` values are passed through.

    Returns
    -------
    list[np.ndarray or None]
        Arrays broadcast to the same shape.
    """
    maxi = 0
    ndim = 0
    for i, arr in enumerate(arrs):
        if arr is None:
            continue
        if arr.ndim > ndim:
            ndim = arr.ndim
            maxi = i
    if ndim < 1:
        raise ValueError("All coordinate arrays are empty or None.")
    shape = arrs[maxi].shape
    reshaped = []
    for arr in arrs:
        if arr is not None and arr.shape != shape:
            if arr.ndim < ndim:
                arr = np.repeat([arr], shape[2 - maxi], axis=2 - maxi)
            else:
                raise ValueError(
                    f"Cannot broadcast coordinate arrays with shapes "
                    f"{[a.shape for a in arrs if a is not None]}."
                )
        reshaped.append(arr)
    return reshaped


def _points(
    self,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    order: str | None = "F",
    scales: dict | None = None,
):
    """Generate structured points as a new interleaved array."""
    if order is None:
        order = "F"
    self._mesh = pv.StructuredGrid()
    ndim = 3 - (x, y, z).count(None)
    if ndim < 2:
        if ndim == 1:
            raise ValueError("One dimensional structured grids should be rectilinear grids.")
        raise ValueError("You must specify at least two dimensions as X, Y, or Z.")
    if x is not None:
        x = self._get_array(x, scale=(scales and scales.get(x)) or 1)
    if y is not None:
        y = self._get_array(y, scale=(scales and scales.get(y)) or 1)
    if z is not None:
        z = self._get_array(z, scale=(scales and scales.get(z)) or 1)
    arrs = _coerce_shapes(x, y, z)
    x, y, z = arrs
    arr = next(a for a in arrs if a is not None)
    points = np.zeros((arr.size, 3), dtype=arr.dtype)
    if x is not None:
        points[:, 0] = x.ravel(order=order)
    if y is not None:
        points[:, 1] = y.ravel(order=order)
    if z is not None:
        points[:, 2] = z.ravel(order=order)
    shape = list(x.shape) + [1] * (3 - ndim)
    return points, shape


def mesh(
    self,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    order: str = "F",
    component: str | None = None,
    scales: dict | None = None,
):
    """Create a :class:`pyvista.StructuredGrid` from curvilinear coordinates.

    Parameters
    ----------
    self : PyVistaAccessor
        The accessor instance (passed internally).
    x, y, z : str, optional
        Names of the coordinate variables for each axis.
        At least two must be specified.
    order : str, default "F"
        Array memory layout for flattening (Fortran order is standard
        for VTK structured grids).
    component : str, optional
        Not currently supported. Raises ``ValueError``.
    scales : dict, optional
        Scale factors for non-numeric coordinates.

    Returns
    -------
    pyvista.StructuredGrid
        The mesh with data values as point data.

    Warns
    -----
    DataCopyWarning
        Always emitted because StructuredGrid creation requires
        copying data into an interleaved point array.

    Notes
    -----
    **Point data vs cell data.** Like all pvxarray mesh builders,
    this method assigns data as **point data** (one value per grid
    node). VTK interpolates point data smoothly across cell faces.

    For curvilinear grids where data represents per-cell averages,
    convert after meshing::

        mesh = mesh.point_data_to_cell_data()

    This averages neighboring node values and is not lossless.
    """
    if order is None:
        order = "F"
    if component is not None:
        raise ValueError("Component is not currently supported for StructuredGrid")
    warnings.warn(
        DataCopyWarning(
            "StructuredGrid accessor duplicates data - VTK/PyVista data not shared with xarray."
        ),
        stacklevel=2,
    )
    points, shape = _points(self, x=x, y=y, z=z, order=order, scales=scales)
    self._mesh.points = points
    self._mesh.dimensions = shape
    data = self.data
    self._mesh[self._obj.name or "data"] = data.ravel(order=order)
    return self._mesh
