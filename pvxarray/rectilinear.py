from typing import Optional
import warnings

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
    self._mesh = pv.RectilinearGrid()
    ndim = 3 - (x, y, z).count(None)
    if ndim < 1:
        raise ValueError("You must specify at least one dimension as X, Y, or Z.")
    # Construct the mesh
    if x is not None:
        self._mesh.x = self._get_array(x)
    if y is not None:
        self._mesh.y = self._get_array(y)
    if z is not None:
        self._mesh.z = self._get_array(z)
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
