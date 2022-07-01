from typing import Optional

import pyvista as pv
import xarray as xr

from pvxarray import rectilinear, structured


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
