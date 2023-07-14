from typing import Optional
import warnings

import numpy as np
import pyvista as pv

from pvxarray.errors import DataCopyWarning


def mesh(
    self,
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    order: Optional[str] = "C",
    component: Optional[str] = None,
):
    if order is None:
        order = "C"
    ndim = 3 - (x, y, z).count(None)
    if ndim < 1:
        raise ValueError("You must specify at least one dimension as X, Y, or Z.")
    values = self.data
    if component is not None:
        # if ndim < values.ndim and values.ndim == ndim + 1:
        # Assuming additional component array
        dims = set(self._obj.dims)
        dims.discard(component)
        values = self._obj.transpose(*dims, component, transpose_coords=True).values
        values = values.reshape((-1, values.shape[-1]), order=order)
        warnings.warn(
            DataCopyWarning(
                "Made a copy of the multicomponent array - VTK/PyVista data not shared with xarray."
            )
        )
        ndim += 1
    else:
        values = values.ravel(order=order)

    # Construct the mesh
    if x is not None:
        x = self._get_array(x)
    else:
        x = np.zeros(values.shape)
    if y is not None:
        y = self._get_array(y)
    else:
        y = np.zeros(values.shape)
    if z is not None:
        z = self._get_array(z)
    else:
        z = np.zeros(values.shape)
    values_dim = len(values)
    # Check dimensionality of data
    if values_dim != len(x):
        raise ValueError(
            f"Dimensional mismatch between specified X, Y, Z coords and dimensionality of DataArray ({len(x)} vs {values_dim})"
        )
    self._mesh = pv.PolyData(np.c_[x, y, z])
    self._mesh[self._obj.name or "data"] = values
    return self._mesh
