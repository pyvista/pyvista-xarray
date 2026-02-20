import json

import numpy as np
import pytest
import xarray as xr

from pvxarray.vtk_source import PyVistaXarraySource, _format_time_labels


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


def test_time_labels_datetime():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")

    labels = source.time_labels
    assert labels is not None
    assert len(labels) == len(da.time)
    # Datetime labels should contain a date-like pattern
    assert "-" in labels[0]
    assert ":" in labels[0]

    # Current time label should match the first entry
    assert source.time_label == labels[0]

    # Changing time_index updates the label
    source.time_index = 1
    assert source.time_label == labels[1]


def test_time_labels_numeric():
    da = xr.DataArray(
        np.zeros((5, 3, 4)),
        dims=["t", "y", "x"],
        coords={"t": np.arange(5), "y": np.arange(3), "x": np.arange(4)},
        name="data",
    )
    source = PyVistaXarraySource(da, x="x", y="y", time="t")

    labels = source.time_labels
    assert labels is not None
    assert labels == ["0", "1", "2", "3", "4"]


def test_time_labels_none_without_time():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air.isel(time=0)
    source = PyVistaXarraySource(da, x="lon", y="lat")

    assert source.time_labels is None
    assert source.time_label is None


def test_format_time_labels_helper():
    # Datetime
    times = xr.DataArray(np.array(["2020-01-01", "2020-06-15"], dtype="datetime64"))
    labels = _format_time_labels(times)
    assert len(labels) == 2
    assert "2020-01-01" in labels[0]

    # Numeric
    times = xr.DataArray(np.array([10, 20, 30]))
    labels = _format_time_labels(times)
    assert labels == ["10", "20", "30"]


def test_multi_array_support():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        z="level",
        time="month",
        dataset=ds,
        arrays=["v", "z"],
    )
    source.time_index = 0
    source.z_index = 0
    mesh = source.apply()

    assert "u" in mesh.point_data
    assert "v" in mesh.point_data
    assert "z" in mesh.point_data


def test_available_arrays():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(da, x="longitude", y="latitude", dataset=ds)

    avail = source.available_arrays
    assert "u" in avail
    assert "v" in avail
    assert "z" in avail


def test_available_arrays_no_dataset():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat")
    assert source.available_arrays == []


def test_arrays_setter():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(da, x="longitude", y="latitude", dataset=ds)

    source.arrays = ["v"]
    assert source.arrays == ["v"]

    source.arrays = []
    assert source.arrays == []


def test_computed_fields():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        z="level",
        time="month",
        dataset=ds,
        arrays=["v"],
    )
    source.time_index = 0
    source.z_index = 0
    source.computed = {
        "_use_scalars": ["u", "v"],
        "speed": "sqrt(u*u + v*v)",
    }
    mesh = source.apply()
    assert "speed" in mesh.point_data
    # Speed should be non-negative
    assert mesh["speed"].min() >= 0


def test_computed_no_fields():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")
    source.computed = {}
    mesh = source.apply()
    assert mesh.n_points > 0


def test_pipeline_callable():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")

    # Simple callable that adds an array
    def add_ones(mesh):
        mesh["ones"] = np.ones(mesh.n_points)
        return mesh

    source.pipeline = [add_ones]
    mesh = source.apply()
    assert "ones" in mesh.point_data
    assert np.allclose(mesh["ones"], 1.0)


def test_pipeline_empty():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")
    source.pipeline = []
    mesh = source.apply()
    assert mesh.n_points > 0


def test_state_roundtrip():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        z="level",
        time="month",
        dataset=ds,
        arrays=["v"],
    )
    source.time_index = 1
    source.resolution = 0.5
    source.computed = {"_use_scalars": ["u", "v"], "speed": "sqrt(u*u + v*v)"}

    state = source.state
    assert state["x"] == "longitude"
    assert state["y"] == "latitude"
    assert state["z"] == "level"
    assert state["time"] == "month"
    assert state["time_index"] == 1
    assert state["resolution"] == 0.5
    assert state["arrays"] == ["v"]
    assert "speed" in state["computed"]


def test_state_to_json_and_back():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    da = ds.u
    source = PyVistaXarraySource(
        da,
        x="longitude",
        y="latitude",
        time="month",
        dataset=ds,
        arrays=["v", "z"],
    )
    source.time_index = 1

    json_str = source.to_json()
    parsed = json.loads(json_str)
    assert parsed["x"] == "longitude"
    assert parsed["arrays"] == ["v", "z"]

    source2 = PyVistaXarraySource.from_json(json_str, data_array=da, dataset=ds)
    assert source2.x == "longitude"
    assert source2.time_index == 1
    assert source2.arrays == ["v", "z"]


def test_load_state():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air
    source = PyVistaXarraySource(da, x="lon", y="lat", time="time")

    state = {"x": "new_x", "y": "new_y", "time_index": 5, "resolution": 0.3}
    source.load_state(state)
    assert source.x == "new_x"
    assert source.y == "new_y"
    assert source.time_index == 5
    assert source.resolution == 0.3
    # Unknown keys should be silently ignored
    source.load_state({"unknown_key": "value"})
