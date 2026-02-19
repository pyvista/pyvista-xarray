"""Tests for the Dataset-level PyVista accessor (ds.pyvista)."""

import numpy as np
import pytest
import pyvista as pv
import xarray as xr

import pvxarray  # noqa: F401


def test_dataset_axes():
    ds = xr.tutorial.load_dataset("air_temperature")
    axes = ds.pyvista.axes
    assert "X" in axes
    assert "Y" in axes
    assert axes["X"] == "lon"
    assert axes["Y"] == "lat"


def test_dataset_axes_eraint():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    axes = ds.pyvista.axes
    assert axes["X"] == "longitude"
    assert axes["Y"] == "latitude"


def test_available_arrays_no_reference():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    avail = ds.pyvista.available_arrays()
    assert "u" in avail
    assert "v" in avail
    assert "z" in avail


def test_available_arrays_with_reference():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    avail = ds.pyvista.available_arrays("u")
    assert "u" in avail
    assert "v" in avail
    assert "z" in avail


def test_available_arrays_excludes_bounds():
    ds = xr.tutorial.load_dataset("air_temperature")
    # Add a fake bounds variable
    ds["lat_bnds"] = xr.DataArray(np.zeros((25, 2)), dims=["lat", "bnds"])
    avail = ds.pyvista.available_arrays()
    assert "lat_bnds" not in avail
    assert "air" in avail


def test_available_arrays_bad_reference():
    ds = xr.tutorial.load_dataset("air_temperature")
    with pytest.raises(KeyError, match="not found"):
        ds.pyvista.available_arrays("nonexistent")


def test_dataset_mesh_single():
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air.isel(time=0)
    ds_2d = da.to_dataset()
    mesh = ds_2d.pyvista.mesh(x="lon", y="lat")
    assert isinstance(mesh, pv.RectilinearGrid)
    assert mesh.n_points > 0


def test_dataset_mesh_multi_array():
    ds = xr.tutorial.load_dataset("eraint_uvz")
    ds_2d = ds.isel(month=0, level=0)
    mesh = ds_2d.pyvista.mesh(arrays=["u", "v", "z"], x="longitude", y="latitude")
    assert "u" in mesh.point_data
    assert "v" in mesh.point_data
    assert "z" in mesh.point_data


def test_dataset_mesh_no_arrays():
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="No data variables"):
        ds.pyvista.mesh()
