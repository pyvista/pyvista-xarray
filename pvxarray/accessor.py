from functools import wraps

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


class BasePyVistaAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

        self._x_coord = None
        self._y_coord = None
        self._z_coord = None

        self._mesh = None

    def _check_safe_dims(self):
        # for dim in self._obj.dims:
        #     if dim not in [self.x_coord, self.y_coord, self.z_coord]:
        #         raise ValueError(f'Please select an index along the `{dim}` dimension.')
        pass

    @property
    def data(self):
        self._check_safe_dims()
        # TODO: check ordering as ZXY (F-ordering)
        return self._obj.values.ravel()

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

    @wraps(pv.plot)
    def plot(self, *args, **kwargs):
        return self.mesh.plot(*args, **kwargs)


@xr.register_dataarray_accessor("pyvista_rectilinear")
class PyVistaRectilinearGridAccessor(BasePyVistaAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        self._x = None
        self._y = None
        self._z = None

    @property
    def x(self):
        return super().x.ravel()

    @property
    def y(self):
        return super().y.ravel()

    @property
    def z(self):
        z = super().z
        if z is not None:
            return z.ravel()

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = pv.RectilinearGrid()
        self._mesh.x = self.x
        self._mesh.y = self.y
        z = self.z
        if z is not None:
            self._mesh.z = self.z
        self._mesh[self._obj.name or "data"] = self.data
        return self._mesh
