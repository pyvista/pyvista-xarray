from typing import Optional
import warnings

import numpy as np
import pyvista as pv
import xarray as xr

from pvxarray import rectilinear, structured
from pvxarray.errors import DataCopyWarning, DataModificationWarning


class _LocIndexer:
    def __init__(self, parent: "PyVistaAccessor"):
        self.parent = parent

    def __getitem__(self, key) -> xr.DataArray:
        return self.parent._obj.loc[key]

    def __setitem__(self, key, value) -> None:
        self.parent._obj.__setitem__(self, key, value)


@xr.register_dataarray_accessor("pyvista")
class PyVistaAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._mesh = None

    def __getitem__(self, key):
        return self._obj.__getitem__(key)

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing like pandas."""
        return _LocIndexer(self)

    def _get_array(self, key):
        try:
            return self._obj[key].values
        except KeyError:
            raise KeyError(
                f"Key {key} not present in DataArray. Choices are: {list(self._obj.coords.keys())}"
            )

    def data(self, nodata: Optional[float] = None):
        values = self._obj.values
        if nodata is not None:
            nans = values == nodata
            if np.any(nans):
                try:
                    values[nans] = np.nan
                    warnings.warn(
                        DataModificationWarning(
                            "nodata values overwritten with `np.nan` in source DataArray."
                        )
                    )
                except ValueError:
                    dytpe = values.dtype
                    values = values.astype(float)
                    values[nans] = np.nan
                    warnings.warn(
                        DataCopyWarning(
                            f"{dytpe} does not support overwritting values with nan. Copying and casting these data to float."
                        )
                    )
        return values

    def mesh(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        order: Optional[str] = None,
        component: Optional[str] = None,
        nodata: Optional[float] = None,
    ) -> pv.DataSet:
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
        if ndim > 1:
            # StructuredGrid
            meth = structured.mesh
        else:
            # RectilinearGrid
            meth = rectilinear.mesh
        return meth(self, x=x, y=y, z=z, order=order, component=component, nodata=nodata)

    def plot(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        order: str = "C",
        nodata: Optional[float] = None,
        **kwargs,
    ):
        return self.mesh(x=x, y=y, z=z, order=order, nodata=nodata).plot(**kwargs)
