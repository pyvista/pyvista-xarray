import numpy as np
import pandas as pd
import pytest
import pyvista as pv
import xarray as xr

import pvxarray


@pytest.fixture
def sample():
    temp = 15 + 8 * np.random.randn(2, 2, 2, 2)
    return xr.Dataset(
        {
            "temperature": (["w", "u", "v", "t"], temp),
        },
        coords={
            "ux": (["u"], np.array([-99.83, -99.32])),
            "uy": (["v"], np.array([42.25, 42.21])),
            "uz": (["w"], np.array([0, 10])),
            "t": (["t"], np.array([0.5, 1.5])),
        },
    ).temperature


def test_accessor_available():
    da = xr.DataArray()
    assert hasattr(da, "pyvista")


def test_indexing(sample):
    ds = sample.pyvista[{"t": 0}]
    assert ds.t == 0.5
    ds = sample.pyvista[{"t": 1}]
    assert ds.t == 1.5
    ds = sample.pyvista.loc[{"t": 0.5}]
    assert ds.t == 0.5
    ds = sample.pyvista.loc[{"t": 1.5}]
    assert ds.t == 1.5


def test_report():
    assert pvxarray.Report()


def test_bad_key(sample):
    with pytest.raises(KeyError):
        sample[{"t": 0}].pyvista.mesh(x="foo")
    with pytest.raises(KeyError):
        sample[{"t": 0}].pyvista.mesh(x="ux", y="hello")
    mesh = sample[{"t": 0}].pyvista.mesh(x="ux", y="uy", z="uz")
    assert mesh.n_points


def test_forgot_choose_time(sample):
    with pytest.raises(ValueError):
        sample.pyvista.mesh(x="ux", y="uy", z="uz")


def test_data_property(sample):
    da = sample[{"t": 0}]
    assert np.array_equal(da.pyvista.data, da.values)


def test_mesh_type_explicit_points():
    lon = np.array([-99.83, -99.32, -98.50])
    lat = np.array([42.25, 42.21, 42.10])
    temp = np.array([15.0, 18.0, 20.0])
    da = xr.DataArray(
        temp,
        dims=["pts"],
        coords={"lon": ("pts", lon), "lat": ("pts", lat)},
        name="temp",
    )
    mesh = da.pyvista.mesh(x="lon", y="lat", mesh_type="points")
    assert isinstance(mesh, pv.PolyData)


def test_mesh_type_explicit_rectilinear():
    lon = np.array([-99.83, -99.32])
    lat = np.array([42.25, 42.21])
    temp = 15 + 8 * np.random.randn(2, 2)
    da = xr.DataArray(
        temp,
        dims=["x", "y"],
        coords={"lon": ("x", lon), "lat": ("y", lat)},
        name="temp",
    )
    mesh = da.pyvista.mesh(x="lon", y="lat", mesh_type="rectilinear")
    assert isinstance(mesh, pv.RectilinearGrid)


def test_mesh_type_explicit_structured():
    x = np.arange(-5, 5, 1.0)
    y = np.arange(-5, 5, 1.0)
    x, y = np.meshgrid(x, y)
    temp = 15 + 8 * np.random.randn(*x.shape)
    da = xr.DataArray(
        temp,
        dims=["xi", "yi"],
        coords={"x": (["xi", "yi"], x), "y": (["xi", "yi"], y)},
        name="temp",
    )
    mesh = da.pyvista.mesh(x="x", y="y", mesh_type="structured")
    assert isinstance(mesh, pv.StructuredGrid)


def test_mesh_type_invalid(sample):
    with pytest.raises(KeyError):
        sample[{"t": 0}].pyvista.mesh(x="ux", y="uy", z="uz", mesh_type="invalid")


def test_get_array_non_numeric():
    times = pd.date_range("2000-01-01", periods=3)
    temp = np.array([15.0, 18.0, 20.0])
    da = xr.DataArray(
        temp,
        dims=["time"],
        coords={"time": times},
        name="temp",
    )
    mesh = da.pyvista.mesh(x="time", mesh_type="rectilinear")
    assert mesh.n_points == 3
    assert np.array_equal(mesh.x, np.array([0, 1, 2]))


def test_loc_setitem():
    da = xr.DataArray(
        np.array([1.0, 2.0, 3.0]),
        dims=["x"],
        coords={"x": [10, 20, 30]},
    )
    da.pyvista.loc[10] = 99.0
    assert da.sel(x=10).item() == 99.0
