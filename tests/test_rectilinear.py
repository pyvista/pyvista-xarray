from pathlib import Path

import numpy as np
import pytest
import rioxarray
import xarray as xr

from pvxarray import DataCopyWarning


@pytest.fixture
def simple():
    lon = np.array([-99.83, -99.32])
    lat = np.array([42.25, 42.21])
    z = np.array([0, 10])
    temp = 15 + 8 * np.random.randn(2, 2, 2)
    ds = xr.Dataset(
        {
            "temperature": (["z", "x", "y"], temp),
        },
        coords={
            "lon": (["x"], lon),
            "lat": (["y"], lat),
            "z": (["z"], z),
        },
    )
    return {"lon": lon, "lat": lat, "z": z, "temp": temp, "ds": ds}


@pytest.fixture
def bahamas_rgb():
    return Path(Path(__file__).parent, "data", "bahamas_rgb.tif").absolute()


def test_simple(simple):
    mesh = simple["ds"].temperature.pyvista.mesh(x="lon", y="lat", z="z")

    assert mesh.n_points == 8
    assert np.array_equal(mesh.x, simple["lon"])
    assert np.array_equal(mesh.y, simple["lat"])
    assert np.array_equal(mesh.z, simple["z"])
    assert np.array_equal(mesh["temperature"], simple["temp"].ravel())


def test_shared_coords(simple):
    mesh = simple["ds"].temperature.pyvista.mesh(x="lon", y="lat", z="z")

    mesh.x[0] = 0
    assert simple["lon"][0] == 0
    assert np.array_equal(mesh.x, simple["lon"])
    assert np.may_share_memory(mesh.x, simple["lon"])

    mesh.y[0] = 0.5
    assert simple["lat"][0] == 0.5
    assert np.array_equal(mesh.y, simple["lat"])
    assert np.may_share_memory(mesh.y, simple["lat"])

    mesh.z[0] = 1
    assert simple["z"][0] == 1
    assert np.array_equal(mesh.z, simple["z"])
    assert np.may_share_memory(mesh.z, simple["z"])


def test_shared_data(simple):
    mesh = simple["ds"].temperature.pyvista.mesh(x="lon", y="lat", z="z")

    mesh["temperature"][0] = -1
    assert simple["temp"].ravel()[0] == -1
    assert np.array_equal(mesh["temperature"], simple["temp"].ravel())
    assert np.may_share_memory(mesh["temperature"], simple["temp"].ravel())


def test_air_temperature():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air[dict(time=0)]

    mesh = da.pyvista.mesh(x="lon", y="lat")
    assert mesh
    assert mesh.n_points == 1325
    assert "air" in mesh.point_data

    assert np.array_equal(mesh["air"], da.values.ravel())
    assert np.may_share_memory(mesh["air"], da.values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    # TODO: why `may_share_memory` failing here?
    # assert np.may_share_memory(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)
    # assert np.may_share_memory(mesh.y, da.lat)


def test_rioxarray(bahamas_rgb):
    da = rioxarray.open_rasterio(bahamas_rgb)
    band = da[dict(band=1)]
    mesh = band.pyvista.mesh(x="x", y="y")
    assert np.array_equal(mesh["data"], band.values.ravel())
    assert np.may_share_memory(mesh["data"], band.values.ravel())
    assert np.array_equal(mesh.x, band.x.values)
    assert np.may_share_memory(mesh.x, band.x.values)
    assert np.array_equal(mesh.y, band.y.values)
    assert np.may_share_memory(mesh.y, band.y.values)


@pytest.mark.xfail(strict=False)
def test_rioxarray_multicomponent(bahamas_rgb):
    da = rioxarray.open_rasterio(bahamas_rgb)
    with pytest.warns(DataCopyWarning):
        mesh = da.pyvista.mesh(x="x", y="y", component="band")
    assert np.array_equal(mesh.x, da.x.values)
    assert np.may_share_memory(mesh.x, da.x.values)
    assert np.array_equal(mesh.y, da.y.values)
    assert np.may_share_memory(mesh.y, da.y.values)
    # Check multicomponent array
    values = da.values.swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 3)
    assert np.allclose(mesh["data"], values)  # TODO: this check is flaky which is very concerning


def test_too_few_dimensions(simple):
    with pytest.raises(ValueError):
        simple["ds"].temperature.pyvista.mesh(x="x")
    with pytest.raises(ValueError):
        mesh = simple["ds"].temperature.pyvista.mesh(x="x", y="y")
        print(mesh["temperature"].shape)


def test_too_many_dimensions(bahamas_rgb):
    da = rioxarray.open_rasterio(bahamas_rgb)
    band = da[dict(band=1)]
    with pytest.raises(ValueError):
        band.pyvista.mesh(x="x", y="y", z="band")


def test_1D_rectilinear_x():
    lon = np.array([-99.83, -99.32, -99.11])
    temp = 15 + 8 * np.random.randn(3)
    ds = xr.Dataset(
        {
            "temperature": (["x"], temp),
        },
        coords={
            "lon": (["x"], lon),
        },
    )
    mesh = ds.temperature.pyvista.mesh(x="lon")
    assert mesh.n_points == 3
    assert np.array_equal(mesh.x, lon)
    assert np.may_share_memory(mesh.x, lon)
    assert np.array_equal(mesh["temperature"], temp)
    assert np.may_share_memory(mesh["temperature"], temp)


def test_1D_rectilinear_y():
    lat = np.array([42.25, 42.21, 42.18])
    temp = 15 + 8 * np.random.randn(3)
    ds = xr.Dataset(
        {
            "temperature": (["y"], temp),
        },
        coords={
            "lat": (["y"], lat),
        },
    )
    mesh = ds.temperature.pyvista.mesh(y="lat")
    assert mesh.n_points == 3
    assert np.array_equal(mesh.y, lat)
    assert np.may_share_memory(mesh.y, lat)
    assert np.array_equal(mesh["temperature"], temp)
    assert np.may_share_memory(mesh["temperature"], temp)


def test_2D_rectilinear_yz():
    lat = np.array([42.25, 42.21])
    z = np.array([0, 10])
    temp = 15 + 8 * np.random.randn(2, 2)
    ds = xr.Dataset(
        {
            "temperature": (["z", "y"], temp),
        },
        coords={
            "lat": (["y"], lat),
            "z": (["z"], z),
        },
    )
    mesh = ds.temperature.pyvista.mesh(y="lat", z="z")
    assert mesh.n_points == 4
    assert np.array_equal(mesh.y, lat)
    assert np.may_share_memory(mesh.y, lat)
    assert np.array_equal(mesh.z, z)
    assert np.may_share_memory(mesh.z, z)
    assert np.array_equal(mesh["temperature"], temp.ravel())
    assert np.may_share_memory(mesh["temperature"], temp)
