import numpy as np
import pytest
import pyvista as pv
import xarray as xr

from pvxarray import DataCopyWarning


@pytest.fixture
def point_data():
    lon = np.array([-99.83, -99.32, -98.50])
    lat = np.array([42.25, 42.21, 42.10])
    alt = np.array([100.0, 200.0, 300.0])
    temp = np.array([15.0, 18.0, 20.0])
    ds = xr.Dataset(
        {
            "temperature": (["pts"], temp),
        },
        coords={
            "lon": (["pts"], lon),
            "lat": (["pts"], lat),
            "alt": (["pts"], alt),
        },
    )
    return {"lon": lon, "lat": lat, "alt": alt, "temp": temp, "ds": ds}


def test_points_mesh_xy(point_data):
    da = point_data["ds"].temperature
    mesh = da.pyvista.mesh(x="lon", y="lat", mesh_type="points")
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 3
    assert np.array_equal(mesh.points[:, 0], point_data["lon"])
    assert np.array_equal(mesh.points[:, 1], point_data["lat"])
    assert np.array_equal(mesh.points[:, 2], np.zeros(3))
    assert np.array_equal(mesh["temperature"], point_data["temp"])


def test_points_mesh_xyz(point_data):
    da = point_data["ds"].temperature
    mesh = da.pyvista.mesh(x="lon", y="lat", z="alt", mesh_type="points")
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 3
    assert np.array_equal(mesh.points[:, 0], point_data["lon"])
    assert np.array_equal(mesh.points[:, 1], point_data["lat"])
    assert np.array_equal(mesh.points[:, 2], point_data["alt"])
    assert np.array_equal(mesh["temperature"], point_data["temp"])


def test_points_mesh_no_coords():
    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["pts"], name="vals")
    # When no explicit coords and no detectable CF axes, auto-detection fails
    with pytest.raises(ValueError, match="Could not auto-detect"):
        da.pyvista.mesh(mesh_type="points")


def test_points_mesh_no_coords_explicit():
    """When x/y/z are all None and passed explicitly, PolyData raises."""
    from pvxarray.points import mesh as points_mesh

    da = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=["pts"], name="vals")
    with pytest.raises(ValueError, match="at least one dimension"):
        points_mesh(da.pyvista, x=None, y=None, z=None)


def test_points_mesh_component():
    lon = np.array([-99.83, -99.32])
    lat = np.array([42.25, 42.21])
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    da = xr.DataArray(
        data,
        dims=["pts", "band"],
        coords={"lon": ("pts", lon), "lat": ("pts", lat)},
        name="rgb",
    )
    with pytest.warns(DataCopyWarning):
        mesh = da.pyvista.mesh(x="lon", y="lat", component="band", mesh_type="points")
    assert isinstance(mesh, pv.PolyData)
    assert mesh.n_points == 2
    assert mesh["rgb"].shape == (2, 3)


def test_points_mesh_dimension_mismatch():
    # 2D data with 1D coords of different lengths causes a mismatch
    temp = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lon = np.array([-99.83, -99.32])
    lat = np.array([42.25, 42.21, 42.10])
    da = xr.DataArray(
        temp,
        dims=["y", "x"],
        coords={"lon": ("x", lon), "lat": ("y", lat)},
        name="temperature",
    )
    # The flattened data has 6 points but lon has 2 and lat has 3
    with pytest.raises(ValueError, match="Dimensional mismatch"):
        da.pyvista.mesh(x="lon", y="lat", mesh_type="points")


def test_points_mesh_data_name():
    lon = np.array([-99.83, -99.32])
    lat = np.array([42.25, 42.21])
    temp = np.array([15.0, 18.0])
    da = xr.DataArray(
        temp,
        dims=["pts"],
        coords={"lon": ("pts", lon), "lat": ("pts", lat)},
    )
    mesh = da.pyvista.mesh(x="lon", y="lat", mesh_type="points")
    assert "data" in mesh.point_data
