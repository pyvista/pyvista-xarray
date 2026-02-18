import numpy as np
import pytest
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

    assert np.array_equal(mesh["air"], da[{"time": 0}].values.ravel())
    # assert np.may_share_memory(mesh["air"], da[dict(time=0)].values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)

    source.time_index = 1
    mesh = source.apply()
    assert np.array_equal(mesh["air"], da[{"time": 1}].values.ravel())
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


def test_source_properties():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")

    assert source.x == "lon"
    assert source.y == "lat"
    assert source.z is None
    assert source.time == "time"
    assert source.order == "C"
    assert source.component is None
    assert source.time_index == 0
    assert source.data_array is da

    source.x = "new_x"
    assert source.x == "new_x"
    source.y = "new_y"
    assert source.y == "new_y"
    source.z = "new_z"
    assert source.z == "new_z"
    source.time = "new_time"
    assert source.time == "new_time"
    source.order = "F"
    assert source.order == "F"
    source.component = "band"
    assert source.component == "band"


def test_source_str():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")
    s = str(source)
    assert "lon" in s
    assert "lat" in s
    assert "time" in s


def test_source_modified_clears_cache():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)

    # Access cached properties to populate them
    _ = source.sliced_data_array
    _ = source.mesh
    assert source._sliced_data_array is not None
    assert source._mesh is not None

    # Modified() should clear all caches
    source.Modified()
    assert source._sliced_data_array is None
    assert source._persisted_data is None
    assert source._mesh is None


def test_source_time_type_error():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    with pytest.raises(TypeError, match="time must be a string or None"):
        PyVistaXarraySource(da, x="lon", y="lat", time=123)


def test_source_z_index():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.z
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        z="level",
        time="month",
    )
    source.time_index = 0
    source.z_index = 0
    assert source.z_index == 0
    sliced = source.sliced_data_array
    assert sliced.ndim == 2


def test_source_data_range():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)
    dmin, dmax = source.data_range
    assert dmin < dmax


def test_source_max_time_index():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")
    assert source.max_time_index == len(da.time) - 1


def test_source_no_data_array():
    source = PyVistaXarraySource()
    assert source.sliced_data_array is None


def test_source_setters_trigger_modified():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time", resolution=1.0)

    # Populate caches
    _ = source.sliced_data_array
    assert source._sliced_data_array is not None

    # data_array setter triggers Modified()
    source.data_array = da
    assert source._sliced_data_array is None

    # Repopulate and test resolution setter
    _ = source.sliced_data_array
    assert source._sliced_data_array is not None
    source.resolution = 0.5
    assert source._sliced_data_array is None
