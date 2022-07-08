from typing import Optional

import pyvista as pv
import xarray as xr

from pvxarray import rectilinear, structured
from pvxarray.cf import get_cf_names


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

    @property
    def data(self):
        return self._obj.values

    def mesh(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        order: Optional[str] = None,
        component: Optional[str] = None,
    ) -> pv.DataSet:
        if (3 - (x, y, z).count(None)) < 1:
            try:
                x, y, z, _ = get_cf_names(self._obj)
            except ImportError:  # pragma: no cover
                pass
        if (3 - (x, y, z).count(None)) < 1:
            raise ValueError(
                "You must specify at least one dimension as X, Y, or Z or install `cf_xarray`."
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
        if ndim > 1:
            # StructuredGrid
            meth = structured.mesh
        else:
            # RectilinearGrid
            meth = rectilinear.mesh
        return meth(self, x=x, y=y, z=z, order=order, component=component)

    def plot(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        order: str = "C",
        **kwargs,
    ):
        return self.mesh(x=x, y=y, z=z, order=order).plot(**kwargs)
