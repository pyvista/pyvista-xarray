import numpy as np
import xarray as xr


def test_air_temperature():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air[dict(time=0)]

    mesh = da.pyvista.mesh()  # No X,Y,Z specified, so should try cf_xarray
    assert mesh
    assert mesh.n_points == 1325
    assert "air" in mesh.point_data

    assert np.array_equal(mesh["air"], da.values.ravel())
    assert np.may_share_memory(mesh["air"], da.values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)
