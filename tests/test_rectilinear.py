from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyvista as pv
from pyvista import ImageData
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
    da = simple["ds"].temperature
    mesh = da.pyvista.mesh(x="lon", y="lat", z="z")

    # Verify mesh coords share memory with what xarray provides
    # (non-dimension coords share with the original numpy array,
    # but dimension coords may not due to pandas Index behavior)
    assert np.may_share_memory(mesh.x, da["lon"].values)
    assert np.may_share_memory(mesh.y, da["lat"].values)
    assert np.may_share_memory(mesh.z, da["z"].values)

    # Verify mutation propagates through the shared memory chain
    mesh.x[0] = 0
    assert da["lon"].values[0] == 0
    assert np.array_equal(mesh.x, da["lon"].values)

    mesh.y[0] = 0.5
    assert da["lat"].values[0] == 0.5
    assert np.array_equal(mesh.y, da["lat"].values)

    mesh.z[0] = 1
    assert da["z"].values[0] == 1
    assert np.array_equal(mesh.z, da["z"].values)


def test_shared_data(simple):
    mesh = simple["ds"].temperature.pyvista.mesh(x="lon", y="lat", z="z")

    mesh["temperature"][0] = -1
    assert simple["temp"].ravel()[0] == -1
    assert np.array_equal(mesh["temperature"], simple["temp"].ravel())
    assert np.may_share_memory(mesh["temperature"], simple["temp"].ravel())


def test_air_temperature():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air[{"time": 0}]

    mesh = da.pyvista.mesh(x="lon", y="lat")
    assert mesh
    assert mesh.n_points == 1325
    assert "air" in mesh.point_data

    assert np.array_equal(mesh["air"], da.values.ravel())
    assert np.may_share_memory(mesh["air"], da.values.ravel())
    assert np.array_equal(mesh.x, da.lon)
    assert np.array_equal(mesh.y, da.lat)


def test_rioxarray(bahamas_rgb):
    da = rioxarray.open_rasterio(bahamas_rgb)
    band = da[{"band": 1}]
    mesh = band.pyvista.mesh(x="x", y="y")
    assert np.array_equal(mesh["data"], band.values.ravel())
    assert np.may_share_memory(mesh["data"], band.values.ravel())
    assert np.array_equal(mesh.x, band.x.values)
    assert np.may_share_memory(mesh.x, band.x.values)
    assert np.array_equal(mesh.y, band.y.values)
    assert np.may_share_memory(mesh.y, band.y.values)


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
    assert np.allclose(mesh["data"], values)


def test_too_few_dimensions(simple):
    with pytest.raises(ValueError):
        simple["ds"].temperature.pyvista.mesh(x="x")
    with pytest.raises(ValueError):
        simple["ds"].temperature.pyvista.mesh(x="x", y="y")


def test_too_many_dimensions(bahamas_rgb):
    da = rioxarray.open_rasterio(bahamas_rgb)
    band = da[{"band": 1}]
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
    da = ds.temperature
    mesh = da.pyvista.mesh(y="lat", z="z")
    assert mesh.n_points == 4
    assert np.array_equal(mesh.y, lat)
    assert np.may_share_memory(mesh.y, da["lat"].values)
    assert np.array_equal(mesh.z, z)
    assert np.may_share_memory(mesh.z, da["z"].values)
    assert np.array_equal(mesh["temperature"], temp.ravel())
    assert np.may_share_memory(mesh["temperature"], temp)


def test_scales():
    times = pd.date_range("2000-01-01", periods=3)
    lat = np.array([42.25, 42.21, 42.10])
    temp = 15 + 8 * np.random.randn(3, 3)
    ds = xr.Dataset(
        {
            "temperature": (["time", "y"], temp),
        },
        coords={
            "time": times,
            "lat": (["y"], lat),
        },
    )
    mesh = ds.temperature.pyvista.mesh(x="time", y="lat", scales={"time": 10.0})
    # Non-numeric "time" coord should be replaced with scaled indices
    assert np.allclose(mesh.x, np.array([0, 10, 20], dtype=float))
    assert np.array_equal(mesh.y, lat)


def test_no_data_name():
    lon = np.array([-99.83, -99.32])
    temp = np.array([15.0, 18.0])
    da = xr.DataArray(
        temp,
        dims=["x"],
        coords={"lon": ("x", lon)},
    )
    mesh = da.pyvista.mesh(x="lon")
    assert "data" in mesh.point_data


# --- ImageData optimization tests ---


@pytest.fixture
def uniform_3d():
    """Create a 3D dataset with uniform spacing on all axes."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    z = np.linspace(0, 3, 4)
    data = np.random.randn(4, 11, 6)
    ds = xr.Dataset(
        {"temperature": (["z", "x", "y"], data)},
        coords={"x": x, "y": y, "z": z},
    )
    return {"x": x, "y": y, "z": z, "data": data, "ds": ds}


def test_uniform_spacing_returns_image_data(uniform_3d):
    """Uniform spacing axes should produce ImageData, not RectilinearGrid."""
    mesh = uniform_3d["ds"].temperature.pyvista.mesh(x="x", y="y", z="z")
    assert isinstance(mesh, ImageData)
    assert not isinstance(mesh, pv.RectilinearGrid)
    assert mesh.n_points == 11 * 6 * 4
    assert mesh.dimensions == (11, 6, 4)
    assert np.allclose(mesh.spacing, (1.0, 1.0, 1.0))
    assert np.allclose(mesh.origin, (0.0, 0.0, 0.0))


def test_uniform_spacing_2d_returns_image_data():
    """2D uniform spacing should also produce ImageData."""
    x = np.linspace(0, 10, 21)
    y = np.linspace(0, 5, 11)
    data = np.random.randn(21, 11)
    ds = xr.Dataset(
        {"temperature": (["x", "y"], data)},
        coords={"x": x, "y": y},
    )
    mesh = ds.temperature.pyvista.mesh(x="x", y="y")
    assert isinstance(mesh, ImageData)
    assert mesh.n_points == 21 * 11
    assert np.allclose(mesh.spacing, (0.5, 0.5, 1.0))


def test_uniform_spacing_1d_returns_image_data():
    """1D uniform spacing should also produce ImageData."""
    x = np.linspace(0, 10, 101)
    data = np.random.randn(101)
    da = xr.DataArray(data, dims=["x"], coords={"x": x}, name="values")
    mesh = da.pyvista.mesh(x="x")
    assert isinstance(mesh, ImageData)
    assert mesh.n_points == 101


def test_nonuniform_spacing_returns_rectilinear_grid():
    """Non-uniform spacing should fall back to RectilinearGrid."""
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])  # non-uniform
    y = np.linspace(0, 5, 6)  # uniform
    data = np.random.randn(5, 6)
    ds = xr.Dataset(
        {"temperature": (["x", "y"], data)},
        coords={"x": x, "y": y},
    )
    mesh = ds.temperature.pyvista.mesh(x="x", y="y")
    assert isinstance(mesh, pv.RectilinearGrid)
    assert not isinstance(mesh, ImageData)


def test_descending_coords_returns_rectilinear_grid():
    """Descending coordinates (negative spacing) should use RectilinearGrid."""
    lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])
    lon = np.linspace(0, 360, 13)
    data = np.random.randn(7, 13)
    ds = xr.Dataset(
        {"temperature": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )
    mesh = ds.temperature.pyvista.mesh(x="lon", y="lat")
    assert isinstance(mesh, pv.RectilinearGrid)


def test_image_data_preserves_data_values(uniform_3d):
    """Data values should be correct on ImageData mesh."""
    mesh = uniform_3d["ds"].temperature.pyvista.mesh(x="x", y="y", z="z")
    assert np.array_equal(mesh["temperature"], uniform_3d["data"].ravel())


def test_image_data_origin_and_spacing():
    """ImageData should have correct origin and spacing."""
    x = np.array([2.0, 4.0, 6.0, 8.0])
    y = np.array([10.0, 13.0, 16.0])
    z = np.array([100.0, 105.0])
    data = np.random.randn(2, 4, 3)
    ds = xr.Dataset(
        {"temp": (["z", "x", "y"], data)},
        coords={"x": x, "y": y, "z": z},
    )
    mesh = ds.temp.pyvista.mesh(x="x", y="y", z="z")
    assert isinstance(mesh, ImageData)
    assert np.allclose(mesh.origin, (2.0, 10.0, 100.0))
    assert np.allclose(mesh.spacing, (2.0, 3.0, 5.0))
    assert mesh.dimensions == (4, 3, 2)


def test_image_data_equivalent_to_rectilinear():
    """ImageData and RectilinearGrid should represent the same geometry."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    data = np.random.randn(11, 6)
    ds = xr.Dataset(
        {"temp": (["x", "y"], data)},
        coords={"x": x, "y": y},
    )

    # Get the ImageData mesh
    im_mesh = ds.temp.pyvista.mesh(x="x", y="y")
    assert isinstance(im_mesh, ImageData)

    # Cast to rectilinear for comparison
    rect = im_mesh.cast_to_rectilinear_grid()
    assert np.allclose(rect.x, x)
    assert np.allclose(rect.y, y)
    assert np.array_equal(rect["temp"], data.ravel())


def test_image_data_with_component():
    """ImageData should work with multi-component arrays."""
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    data = np.random.randn(3, 11, 6)
    da = xr.DataArray(
        data,
        dims=["band", "x", "y"],
        coords={"x": x, "y": y, "band": [1, 2, 3]},
        name="rgb",
    )
    with pytest.warns(DataCopyWarning):
        mesh = da.pyvista.mesh(x="x", y="y", component="band")
    assert isinstance(mesh, ImageData)
    assert mesh.n_points == 11 * 6


def test_image_data_roundtrip():
    """ImageData → xarray → ImageData should roundtrip cleanly."""
    from pvxarray import pyvista_to_xarray

    x = np.linspace(0, 10, 6)
    y = np.linspace(0, 5, 4)
    z = np.linspace(0, 2, 3)
    data = np.random.randn(3, 6, 4)
    ds = xr.Dataset(
        {"data_var": (["z", "x", "y"], data)},
        coords={"x": x, "y": y, "z": z},
    )

    # Create ImageData
    mesh1 = ds.data_var.pyvista.mesh(x="x", y="y", z="z")
    assert isinstance(mesh1, ImageData)

    # Convert back to xarray
    ds2 = pyvista_to_xarray(mesh1)

    # Create mesh again
    mesh2 = ds2["data_var"].pyvista.mesh(x="x", y="y", z="z")
    assert isinstance(mesh2, ImageData)
    assert np.allclose(mesh1.origin, mesh2.origin)
    assert np.allclose(mesh1.spacing, mesh2.spacing)
    assert mesh1.dimensions == mesh2.dimensions


def test_cells3d_produces_image_data():
    """The cells3d tutorial dataset should produce ImageData (uniform spacing)."""
    ds = xr.tutorial.load_dataset("cells3d")
    da = ds.images.sel(c="nuclei")
    mesh = da.pyvista.mesh(x="x", y="y", z="z")
    assert isinstance(mesh, ImageData)
    assert mesh.n_points == len(da.x) * len(da.y) * len(da.z)
    assert "images" in mesh.point_data
