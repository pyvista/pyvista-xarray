def get_cf_names(da):
    """Use `cf_xarray` to get the names of the X, Y, and Z arrays."""
    try:
        import cf_xarray  # noqa

        axes = da.cf.axes
    except (AttributeError, ImportError):  # pragma: no cover
        raise ImportError("Please install `cf_xarray` to use CF conventions.")
    x = axes.get("X", [None])[0]
    y = axes.get("Y", [None])[0]
    z = axes.get("Z", [None])[0]
    t = axes.get("T", [None])[0]
    return x, y, z, t
