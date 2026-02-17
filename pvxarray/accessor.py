import numpy as np
import pyvista as pv
import xarray as xr

from pvxarray import points, rectilinear, structured
from pvxarray.vtk_source import PyVistaXarraySource

methods = {
    "points": points.mesh,
    "rectilinear": rectilinear.mesh,
    "structured": structured.mesh,
}


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

    def _get_array(self, key, scale=1):
        try:
            values = self._obj[key].values
            if "float" not in str(values.dtype) and "int" not in str(values.dtype):
                # non-numeric coordinate, assign array of scaled indices
                values = np.array(range(len(values))) * scale
            return values
        except KeyError as e:
            raise KeyError(
                f"Key {key} not present in DataArray. Choices are: {list(self._obj.coords.keys())}"
            ) from e

    @property
    def data(self):
        return self._obj.values

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
        if mesh_type is None:  # Try to guess mesh type
            mesh_type = "structured" if ndim > 1 else "rectilinear"
        try:
            meth = methods[mesh_type]
        except KeyError as e:
            raise KeyError from e
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
    ):
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
