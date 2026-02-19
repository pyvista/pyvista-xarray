"""Create PyVista PolyData (point cloud) meshes from xarray DataArrays.

PolyData is used for scattered/unstructured point data where
coordinates are 1D arrays of the same length (one value per point).
This is appropriate for observation data, station networks, or any
data without a regular grid structure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyvista as pv

from pvxarray.errors import DataCopyWarning


def mesh(
    self,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    order: str | None = "C",
    component: str | None = None,
    scales: dict | None = None,
):
    """Create a :class:`pyvista.PolyData` point cloud from coordinates.

    Parameters
    ----------
    self : PyVistaAccessor
        The accessor instance (passed internally).
    x, y, z : str, optional
        Names of the coordinate variables for each axis.
        At least one must be specified. Missing axes are filled
        with zeros.
    order : str, default "C"
        Array memory layout for flattening data values.
    component : str, optional
        Name of an extra dimension for multi-component arrays
        (e.g. ``"band"`` for RGB data). Triggers a data copy.
    scales : dict, optional
        Scale factors for non-numeric coordinates.

    Returns
    -------
    pyvista.PolyData
        A point cloud mesh with data values as point data.
    """
    if order is None:
        order = "C"
    ndim = 3 - (x, y, z).count(None)
    if ndim < 1:
        raise ValueError("You must specify at least one dimension as X, Y, or Z.")
    values = self.data
    if component is not None:
        dims = set(self._obj.dims)
        dims.discard(component)
        values = self._obj.transpose(*dims, component, transpose_coords=True).values
        values = values.reshape((-1, values.shape[-1]), order=order)
        warnings.warn(
            DataCopyWarning(
                "Made a copy of the multicomponent array - VTK/PyVista data not shared with xarray."
            ),
            stacklevel=2,
        )
        ndim += 1
    else:
        values = values.ravel(order=order)

    # Construct the mesh
    n_points = len(values)
    x = (
        self._get_array(x, scale=(scales and scales.get(x)) or 1)
        if x is not None
        else np.zeros(n_points)
    )
    y = (
        self._get_array(y, scale=(scales and scales.get(y)) or 1)
        if y is not None
        else np.zeros(n_points)
    )
    z = (
        self._get_array(z, scale=(scales and scales.get(z)) or 1)
        if z is not None
        else np.zeros(n_points)
    )
    values_dim = len(values)
    # Check dimensionality of data
    if values_dim != len(x):
        raise ValueError(
            f"Dimensional mismatch between specified X, Y, Z coords "
            f"and dimensionality of DataArray ({len(x)} vs {values_dim})"
        )
    self._mesh = pv.PolyData(np.c_[x, y, z])
    self._mesh[self._obj.name or "data"] = values
    return self._mesh
