from functools import wraps

import numpy as np
import pyvista as pv
import xarray as xr

from pvxarray import knowledge


def _get_spatial_coords(data_array):
    coords = data_array.coords  # TODO: handle case
    x_coord, y_coord = None, None
    for xc, yc in knowledge.XY_NAMES:
        if xc in coords and yc in coords:
            x_coord = xc
            y_coord = yc
    if x_coord is None or y_coord is None:
        raise KeyError(f"Spatial coordinates not understood: {data_array.coords}")
    return x_coord, y_coord


def _get_z_coord(data_array):
    for var in data_array.coords:
        if var.lower() in knowledge.Z_NAMES:
            return var
    return None


class _LocIndexer:
    def __init__(self, parent: "BasePyVistaAccessor"):
        self.parent = parent

    def __getitem__(self, key) -> xr.DataArray:
        result = self.parent._obj.loc[key]
        if isinstance(self.parent, PyVistaRectilinearGridAccessor):
            result.pyvista.copy_meta(self.parent)
            result.pyvista.update()
        elif isinstance(self.parent, PyVistaStructuredGridAccessor):
            result.pyvista_structured.copy_meta(self.parent)
            result.pyvista_structured.update()
        return result

    def __setitem__(self, key, value) -> None:
        self.parent._obj.__setitem__(self, key, value)


class BasePyVistaAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

        self._x_coord = None
        self._y_coord = None
        self._z_coord = None

        self._mesh = None

    @staticmethod
    def _copy_meta(output, source):
        output._x_coord = source._x_coord
        output._y_coord = source._y_coord
        output._z_coord = source._z_coord

        output._mesh = source._mesh

    def copy_meta(self, source):
        BasePyVistaAccessor._copy_meta(self, source)

    def __getitem__(self, key):
        return self._obj.__getitem__(key)

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing like pandas."""
        return _LocIndexer(self)

    def _check_safe_dims(self):
        # TODO: check ordering as ZXY, not just shape
        # for dim in self._obj.dims:
        #     if dim not in [self.x_coord, self.y_coord, self.z_coord]:
        #         raise ValueError(f'Please select an index along the `{dim}` dimension.')
        pass

    @property
    def data(self):
        self._check_safe_dims()
        return self._obj.values

    @property
    def x_coord(self):
        if self._x_coord is None:
            try:
                x_coord, _ = _get_spatial_coords(self._obj)
                self._x_coord = x_coord
                return x_coord
            except KeyError:
                pass
        return self._x_coord

    @x_coord.setter
    def x_coord(self, name):
        if name not in self._obj.coords:
            raise KeyError(f"{name} not found in coords")
        self._x_coord = name

    @property
    def y_coord(self):
        if self._y_coord is None:
            try:
                _, y_coord = _get_spatial_coords(self._obj)
                self._y_coord = y_coord
                return y_coord
            except KeyError:
                pass
        return self._y_coord

    @y_coord.setter
    def y_coord(self, name):
        if name not in self._obj.coords:
            raise KeyError(f"{name} not found in coords")
        self._y_coord = name

    @property
    def z_coord(self):
        if self._z_coord is None:
            try:
                z_coord = _get_z_coord(self._obj)
                self._z_coord = z_coord
                return z_coord
            except KeyError:
                pass
        return self._z_coord

    @z_coord.setter
    def z_coord(self, name):
        if name not in self._obj.coords:
            raise KeyError(f"{name} not found in coords")
        self._z_coord = name

    @property
    def x(self):
        if self._x is not None:
            return self._x
        if self.x_coord is None:
            raise ValueError(f"x coord not set. Please set to one of: {self._obj.coords}")
        self._x = self._obj[self.x_coord].values
        return self._x

    @property
    def y(self):
        if self._y is not None:
            return self._y
        if self.y_coord is None:
            raise ValueError(f"y coord not set. Please set to one of: {self._obj.coords}")
        self._y = self._obj[self.y_coord].values
        return self._y

    @property
    def z(self):
        if self._z is not None:
            return self._z
        if self.z_coord is not None:
            self._z = self._obj[self.z_coord].values
        return self._z

    @property
    def mesh(self):
        raise NotImplementedError

    def update(self):
        self.mesh  # fetch mesh so values are updated

    @wraps(pv.plot)
    def plot(self, *args, **kwargs):
        return self.mesh.plot(*args, **kwargs)


@xr.register_dataarray_accessor("pyvista")
class PyVistaRectilinearGridAccessor(BasePyVistaAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self._x = None
        self._y = None
        self._z = None

        self._mesh = pv.RectilinearGrid()

    def __getitem__(self, key):
        result = super().__getitem__(key)
        result.pyvista.copy_meta(self)
        result.pyvista.update()
        return result

    @property
    def mesh(self):
        self._mesh.x = self.x
        self._mesh.y = self.y
        z = self.z
        if z is not None:
            self._mesh.z = self.z
        self._mesh[self._obj.name or "data"] = self.data.ravel()
        return self._mesh


@xr.register_dataarray_accessor("pyvista_structured")
class PyVistaStructuredGridAccessor(BasePyVistaAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self._x = None
        self._y = None
        self._z = None

        self._mesh = pv.StructuredGrid()

    def __getitem__(self, key):
        result = super().__getitem__(key)
        result.pyvista_structured.copy_meta(self)
        result.pyvista_structured.update()
        return result

    @property
    def x(self):
        x = super().x
        if x.ndim < 3 and self.z is not None:
            x = np.repeat([x], self.z.shape[0], axis=0)
        return x

    @property
    def y(self):
        y = super().y
        if y.ndim < 3 and self.z is not None:
            y = np.repeat([y], self.z.shape[0], axis=0)
        return y

    @property
    def points(self):
        """Generate structured points as new array."""
        points = np.zeros((self.x.size, 3), self.x.dtype)
        points[:, 0] = self.x.ravel(order="F")
        points[:, 1] = self.y.ravel(order="F")
        if self.z is not None:
            points[:, 2] = self.z.ravel(order="F")
        return points

    @property
    def data(self):
        v = super().data
        if v.shape != self.x.shape:
            raise ValueError(
                "Coord and data shape mismatch. You may need to `transpose` the DataArray."
            )
        return v

    @property
    def mesh(self):
        self._mesh.points = self.points
        shape = self.x.shape
        self._mesh.dimensions = list(shape) + [1] * (3 - len(shape))
        self._mesh[self._obj.name or "data"] = self.data.ravel(order="F")
        return self._mesh
