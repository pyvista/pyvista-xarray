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


def test_vtk_source_time_as_spatial():
    ds = xr.tutorial.load_dataset("air_temperature")

    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", z="time")

    mesh = source.apply()
    assert mesh
    assert mesh.n_points == 3869000
    assert "air" in mesh.point_data

    assert np.array_equal(mesh["air"], da.values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)
    # Z values are indexes instead of datetime objects
    assert np.array_equal(mesh.z, list(range(da.time.size)))


def test_vtk_source_slicing():
    ds = xr.tutorial.load_dataset("eraint_uvz")

    da = ds.z
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        z="level",
        time="month",
    )
    source.time_index = 1
    source.slicing = {
        "latitude": [0, 241, 2],
        "longitude": [0, 480, 4],
        "level": [0, 3, 1],
        "month": [0, 2, 1],  # should be ignored in favor of t_index
    }

    sliced = source.sliced_data_array
    assert sliced.shape == (3, 121, 120)
