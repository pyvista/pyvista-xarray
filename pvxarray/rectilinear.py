from typing import Dict, Optional
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
    scales: Optional[Dict] = None,
) -> pv.RectilinearGrid | pv.ImageData:
    if order is None:
        order = "C"

    ndim = 3 - (x, y, z).count(None)
    if ndim < 1:
        raise ValueError("You must specify at least one dimension as X, Y, or Z.")
    # Construct the mesh
    if x is not None:
        xx = self._get_array(x, scale=(scales and scales.get(x)) or 1)
    else:
        xx = np.array([0.0])
    if y is not None:
        yy = self._get_array(y, scale=(scales and scales.get(y)) or 1)
    else:
        yy = np.array([0.0])
    if z is not None:
        zz = self._get_array(z, scale=(scales and scales.get(z)) or 1)
    else:
        zz = np.array([0.0])

    dx = np.diff(xx)
    dy = np.diff(yy)
    dz = np.diff(zz)

    ddx = dx[0] if len(dx) and dx[0] > 0 else 1.0
    ddy = dy[0] if len(dy) and dy[0] > 0 else 1.0
    ddz = dz[0] if len(dz) and dz[0] > 0 else 1.0

    if np.allclose(dx, ddx) and np.allclose(dy, ddy) and np.allclose(dz, ddz):
        self._mesh = pv.ImageData(
            origin=(xx[0], yy[0], zz[0]),
            spacing=(ddx, ddy, ddz),
            dimensions=(len(xx), len(yy), len(zz)),
        )
    else:
        self._mesh = pv.RectilinearGrid()
        self._mesh.x = xx
        self._mesh.y = yy
        self._mesh.z = zz

    # Handle data values
    values = self.data
    values_dim = values.ndim
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
    # Check dimensionality of data
    if values_dim != ndim:
        msg = f"Dimensional mismatch between specified X, Y, Z coords and dimensionality of DataArray ({ndim} vs {values_dim})"
        if ndim > values_dim:
            raise ValueError(
                f"{msg}. Too many coordinate dimensions specified leave out Y and/or Z."
            )
        raise ValueError(
            f"{msg}. Too few coordinate dimensions specified. Be sure to specify Y and/or Z or reduce the dimensionality of the DataArray by indexing along non-spatial coordinates like Time."
        )
    self._mesh[self._obj.name or "data"] = values
    return self._mesh
