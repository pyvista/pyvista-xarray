import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def simple():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.sin(np.sqrt(x**2 + y**2))
    x, y, z = np.meshgrid(x, y, z)
    temp = 15 + 8 * np.random.randn(*x.shape)

    temp.shape
    da = xr.DataArray(temp, coords=[z, x, y], name="temperature")
    return {"lon": x, "lat": y, "z": z, "temp": temp, "da": da}


def test_simple(simple):
    mesh = simple["da"].pyvista_structured.mesh

    assert mesh.n_points == simple["lon"].size
    assert np.allclose(mesh.x, simple["lon"])
    assert np.allclose(mesh.y, simple["lat"])
    assert np.allclose(mesh.z, simple["z"])
    assert np.allclose(mesh["temperature"], simple["temp"].ravel())
