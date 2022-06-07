XY_NAMES = {
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("easting", "northing"),
    ("east", "north"),
    ("xc", "yc"),
    ("x", "y"),
}

Z_NAMES = {"altitude", "depth", "zc", "z"}


def register_xy_names(x, y):
    if not isinstance(x, str) or not isinstance(y, str):
        raise TypeError(f"x and y must both be of str type, not: ({type(x)}, {type(y)})")
    XY_NAMES.add((x, y))


def register_z_name(z):
    if not isinstance(z, str):
        raise TypeError(f"z must be of str type, not: {type(z)}")
    Z_NAMES.add(z)


def register_coord_names(x, y, z):
    register_xy_names(x, y)
    register_z_name(z)
