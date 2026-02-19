"""Tests for CF-convention coordinate auto-detection via cf-xarray."""

import numpy as np
import pytest
import pyvista as pv
import xarray as xr

import pvxarray  # noqa: F401
from pvxarray.cf import detect_axes

# ---------------------------------------------------------------------------
# detect_axes — integration tests
# ---------------------------------------------------------------------------


class TestDetectAxes:
    """Test the detect_axes wrapper around cf-xarray."""

    def test_name_heuristics(self):
        """Coordinates named 'lat' and 'lon' should be detected."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["lat", "lon"],
            coords={"lat": np.arange(3), "lon": np.arange(4)},
        )
        axes = detect_axes(da)
        assert axes["X"] == "lon"
        assert axes["Y"] == "lat"

    def test_explicit_axis_attributes(self):
        """Coordinates with axis='X'/'Y'/'Z' attributes should be detected."""
        da = xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=["z_dim", "y_dim", "x_dim"],
            coords={
                "x_coord": ("x_dim", np.arange(4), {"axis": "X"}),
                "y_coord": ("y_dim", np.arange(3), {"axis": "Y"}),
                "z_coord": ("z_dim", np.arange(2), {"axis": "Z"}),
            },
        )
        axes = detect_axes(da)
        assert axes["X"] == "x_coord"
        assert axes["Y"] == "y_coord"
        assert axes["Z"] == "z_coord"

    def test_units_based_detection(self):
        """Coordinates with CF units (degrees_east/north) should be detected."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["y_dim", "x_dim"],
            coords={
                "x_coord": ("x_dim", np.arange(4), {"units": "degrees_east"}),
                "y_coord": ("y_dim", np.arange(3), {"units": "degrees_north"}),
            },
        )
        axes = detect_axes(da)
        assert axes["X"] == "x_coord"
        assert axes["Y"] == "y_coord"

    def test_standard_name_detection(self):
        """Coordinates with standard_name attributes should be detected."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["y_dim", "x_dim"],
            coords={
                "x_coord": ("x_dim", np.arange(4), {"standard_name": "longitude"}),
                "y_coord": ("y_dim", np.arange(3), {"standard_name": "latitude"}),
            },
        )
        axes = detect_axes(da)
        assert axes["X"] == "x_coord"
        assert axes["Y"] == "y_coord"

    def test_no_detectable_axes(self):
        """DataArray with non-standard coordinate names returns empty dict."""
        da = xr.DataArray(
            np.zeros(5),
            dims=["foo"],
            coords={"foo": np.arange(5)},
        )
        axes = detect_axes(da)
        assert axes == {}

    def test_scalar_coords_skipped(self):
        """Scalar (0-dim) coordinates should be ignored.

        After .sel(level=N), 'level' becomes a 0-dim coord and should
        not be detected as a spatial axis.
        """
        da = xr.DataArray(
            np.zeros((3, 4, 2)),
            dims=["lat", "lon", "level"],
            coords={
                "lat": np.arange(3),
                "lon": np.arange(4),
                "level": np.arange(2),
            },
        )
        # Before selection: level should be detected
        axes_3d = detect_axes(da)
        assert "Z" in axes_3d

        # After selection: level becomes scalar, should not be detected
        da_2d = da.sel(level=0)
        axes_2d = detect_axes(da_2d)
        assert "Z" not in axes_2d
        assert axes_2d["X"] == "lon"
        assert axes_2d["Y"] == "lat"

    def test_air_temperature_dataset(self):
        """Integration test with xarray tutorial data."""
        ds = xr.tutorial.load_dataset("air_temperature")
        da = ds.air.isel(time=0)
        axes = detect_axes(da)
        assert axes["Y"] == "lat"
        assert axes["X"] == "lon"

    def test_eraint_uvz_dataset(self):
        """Integration test with ERA-Interim data (units-based detection)."""
        ds = xr.tutorial.load_dataset("eraint_uvz")
        da = ds.z.isel(month=0)
        axes = detect_axes(da)
        assert axes["X"] == "longitude"
        assert axes["Y"] == "latitude"
        assert axes["Z"] == "level"


# ---------------------------------------------------------------------------
# PyVistaAccessor auto-detection integration
# ---------------------------------------------------------------------------


class TestAccessorAutoDetect:
    """Test that accessor.mesh() auto-detects coordinates via cf-xarray."""

    def test_mesh_auto_detect_air_temperature(self):
        """da.pyvista.mesh() without explicit x/y should work for air_temperature."""
        ds = xr.tutorial.load_dataset("air_temperature")
        da = ds.air.isel(time=0)
        mesh = da.pyvista.mesh()
        assert isinstance(mesh, pv.RectilinearGrid)
        assert mesh.n_points > 0

    def test_mesh_auto_detect_with_cf_attributes(self):
        """Auto-detection from explicit axis attrs."""
        da = xr.DataArray(
            np.random.randn(3, 4),
            dims=["y_dim", "x_dim"],
            coords={
                "xc": ("x_dim", np.arange(4), {"axis": "X"}),
                "yc": ("y_dim", np.arange(3), {"axis": "Y"}),
            },
            name="temp",
        )
        mesh = da.pyvista.mesh()
        assert isinstance(mesh, pv.RectilinearGrid)
        assert mesh.n_points == 12

    def test_mesh_auto_detect_fails_gracefully(self):
        """Auto-detection raises ValueError when no axes found."""
        da = xr.DataArray(
            np.zeros(5),
            dims=["foo"],
            coords={"foo": np.arange(5)},
            name="data",
        )
        with pytest.raises(ValueError, match="Could not auto-detect"):
            da.pyvista.mesh()

    def test_axes_property(self):
        """Test the .pyvista.axes property."""
        ds = xr.tutorial.load_dataset("air_temperature")
        da = ds.air.isel(time=0)
        axes = da.pyvista.axes
        assert isinstance(axes, dict)
        assert "X" in axes
        assert "Y" in axes

    def test_spatial_coords_property(self):
        """Test the .pyvista.spatial_coords property."""
        ds = xr.tutorial.load_dataset("air_temperature")
        da = ds.air.isel(time=0)
        coords = da.pyvista.spatial_coords
        assert isinstance(coords, list)
        assert len(coords) >= 2  # at least lon, lat

    def test_explicit_coords_override_auto_detect(self):
        """Specifying x/y explicitly should bypass auto-detection."""
        da = xr.DataArray(
            np.random.randn(3, 4),
            dims=["a", "b"],
            coords={"a": np.arange(3), "b": np.arange(4)},
            name="data",
        )
        # These coords have non-standard names, so auto-detect would fail.
        # But explicit specification should work.
        mesh = da.pyvista.mesh(x="b", y="a")
        assert isinstance(mesh, pv.RectilinearGrid)
        assert mesh.n_points == 12
