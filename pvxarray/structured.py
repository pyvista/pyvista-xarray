import warnings

import numpy as np
import pyvista as pv

from pvxarray.errors import DataCopyWarning


def _coerce_shapes(*arrs):
    """Coerce all argument arrays to have the same shape."""
    maxi = 0
    ndim = 0
    for i, arr in enumerate(arrs):
        if arr is None:
            continue
        if arr.ndim > ndim:
            ndim = arr.ndim
            maxi = i
    # print(arrs)
    # if ndim != len(arrs) - (*arrs,).count(None):
    #     print(ndim, len(arrs))
    #     raise ValueError
    if ndim < 1:
        raise ValueError
    shape = arrs[maxi].shape
    reshaped = []
    for arr in arrs:
        if arr is not None and arr.shape != shape:
            if arr.ndim < ndim:
                arr = np.repeat([arr], shape[2 - maxi], axis=2 - maxi)
            else:
                raise ValueError
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
    """Generate structured points as new array."""
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
    component: str | None = None,  # TODO
    scales: dict | None = None,
):
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
    # if tuple(data.shape) != tuple(shape):
    #     raise ValueError(
    #         "Coord and data shape mismatch. You may need to `transpose` the DataArray. "
    #         f"Data shape {data.shape} vs. mesh shape {shape}"
    #     )
    self._mesh[self._obj.name or "data"] = data.ravel(order=order)
    return self._mesh
