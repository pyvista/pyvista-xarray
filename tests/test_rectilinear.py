import numpy as np
import pytest
import xarray as xr


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


def test_simple(simple):
    mesh = simple["ds"].temperature.pyvista_rectilinear.mesh

    assert mesh.n_points == 8
    assert np.allclose(mesh.x, simple["lon"])
    assert np.allclose(mesh.y, simple["lat"])
    assert np.allclose(mesh.z, simple["z"])
    assert np.allclose(mesh["temperature"], simple["temp"].ravel())


def test_shared_coords(simple):
    mesh = simple["ds"].temperature.pyvista_rectilinear.mesh

    mesh.x[0] = 0
    assert simple["lon"][0] == 0
    assert np.allclose(mesh.x, simple["lon"])

    mesh.y[0] = 0.5
    assert simple["lat"][0] == 0.5
    assert np.allclose(mesh.y, simple["lat"])

    mesh.z[0] = 1
    assert simple["z"][0] == 1
    assert np.allclose(mesh.z, simple["z"])


def test_shared_data(simple):
    mesh = simple["ds"].temperature.pyvista_rectilinear.mesh

    mesh["temperature"][0] = -1
    assert simple["temp"].ravel()[0] == -1
    assert np.allclose(mesh["temperature"], simple["temp"].ravel())


def test_air_temperature():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air[dict(time=0)]

    mesh = da.pyvista_rectilinear.mesh
    assert mesh
    assert mesh.n_points == 1325
    assert "air" in mesh.point_data

    assert np.allclose(mesh["air"], da.values.ravel())
    assert np.allclose(mesh.x, da.lon)
    assert np.allclose(mesh.y, da.lat)
