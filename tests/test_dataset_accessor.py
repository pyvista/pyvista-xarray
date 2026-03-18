"""Tests for the Dataset-level PyVista accessor (ds.pyvista)."""

import numpy as np
import pytest
import pyvista as pv
import xarray as xr

import pvxarray  # noqa: F401
from pvxarray.vtk_source import PyVistaXarraySource


def test_dataset_axes():
    """CF axis detection should work on the air_temperature Dataset."""
    ds = xr.tutorial.load_dataset("air_temperature")
    axes = ds.pyvista.axes
    assert "X" in axes
    assert "Y" in axes
    assert axes["X"] == "lon"
    assert axes["Y"] == "lat"


def test_dataset_axes_eraint():
    """CF axis detection should work on the eraint_uvz Dataset."""
    ds = xr.tutorial.load_dataset("eraint_uvz")
    axes = ds.pyvista.axes
    assert axes["X"] == "longitude"
    assert axes["Y"] == "latitude"


def test_available_arrays_no_reference():
    """All non-bounds data variables should be returned without a reference."""
    ds = xr.tutorial.load_dataset("eraint_uvz")
    avail = ds.pyvista.available_arrays()
    assert "u" in avail
    assert "v" in avail
    assert "z" in avail


def test_available_arrays_with_reference():
    """Variables sharing dimensions with the reference should be returned."""
    ds = xr.tutorial.load_dataset("eraint_uvz")
    avail = ds.pyvista.available_arrays("u")
    assert "u" in avail
    assert "v" in avail
    assert "z" in avail


def test_available_arrays_excludes_bounds():
    """CF boundary variables should be excluded from available arrays."""
    ds = xr.tutorial.load_dataset("air_temperature")
    ds["lat_bnds"] = xr.DataArray(np.zeros((25, 2)), dims=["lat", "bnds"])
    avail = ds.pyvista.available_arrays()
    assert "lat_bnds" not in avail
    assert "air" in avail


def test_available_arrays_bad_reference():
    """A missing reference variable should raise KeyError."""
    ds = xr.tutorial.load_dataset("air_temperature")
    with pytest.raises(KeyError, match="not found"):
        ds.pyvista.available_arrays("nonexistent")


def test_dataset_mesh_single():
    """A single-variable Dataset should produce a valid mesh."""
    ds = xr.tutorial.load_dataset("air_temperature")
    da = ds.air.isel(time=0)
    ds_2d = da.to_dataset()
    mesh = ds_2d.pyvista.mesh(x="lon", y="lat")
    assert isinstance(mesh, pv.RectilinearGrid)
    assert mesh.n_points > 0


def test_dataset_mesh_multi_array():
    """Multiple arrays should be loaded as separate point data."""
    ds = xr.tutorial.load_dataset("eraint_uvz")
    ds_2d = ds.isel(month=0, level=0)
    mesh = ds_2d.pyvista.mesh(arrays=["u", "v", "z"], x="longitude", y="latitude")
    assert "u" in mesh.point_data
    assert "v" in mesh.point_data
    assert "z" in mesh.point_data


def test_dataset_mesh_no_arrays():
    """An empty Dataset should raise ValueError."""
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="No data variables"):
        ds.pyvista.mesh()


def test_dataset_algorithm():
    """Algorithm source should load multiple arrays from the Dataset."""
    ds = xr.tutorial.load_dataset("eraint_uvz")
    source = ds.pyvista.algorithm(
        arrays=["u", "v"],
        x="longitude",
        y="latitude",
        z="level",
        time="month",
    )
    assert isinstance(source, PyVistaXarraySource)
    assert source.arrays == ["v"]  # u is primary, v is extra
    assert source.dataset is ds

    source.time_index = 0
    source.z_index = 0
    mesh = source.apply()
    assert "u" in mesh.point_data
    assert "v" in mesh.point_data


def test_dataset_algorithm_no_arrays():
    """An empty Dataset should raise ValueError for algorithm too."""
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="No data variables"):
        ds.pyvista.algorithm()


def test_dataset_algorithm_default_arrays():
    """Algorithm should select the first data variable when arrays is None."""
    ds = xr.tutorial.load_dataset("air_temperature")
    source = ds.pyvista.algorithm(x="lon", y="lat", time="time")
    assert isinstance(source, PyVistaXarraySource)
    assert source.data_array.name == "air"
