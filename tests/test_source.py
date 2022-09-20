import numpy as np
import xarray as xr

from pvxarray.vtk_source import PyVistaXarraySource


def test_vtk_source():
    ds = xr.tutorial.load_dataset("air_temperature")

    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)

    mesh = source.apply()
    assert mesh
    assert mesh.n_points == 1325
    assert "air" in mesh.point_data

    assert np.array_equal(mesh["air"], da[dict(time=0)].values.ravel())
    # assert np.may_share_memory(mesh["air"], da[dict(time=0)].values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)

    source.time_index = 1
    mesh = source.apply()
    assert np.array_equal(mesh["air"], da[dict(time=1)].values.ravel())
    # assert np.may_share_memory(mesh["air"], da[dict(time=1)].values.ravel())

    source.resolution = 0.5
    mesh = source.apply()
    assert mesh.n_points < 1325
