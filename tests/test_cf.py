"""Tests for CF-convention coordinate auto-detection."""

import numpy as np
import pytest
import pyvista as pv
import xarray as xr

import pvxarray  # noqa: F401
from pvxarray.cf import _detect_axis_for_coord, detect_axes

# ---------------------------------------------------------------------------
# _detect_axis_for_coord — Priority 1: axis attribute
# ---------------------------------------------------------------------------


class TestDetectByAxisAttr:
    """Test detection via the ``axis`` coordinate attribute."""

    @pytest.mark.parametrize(
        "axis_val,expected",
        [
            ("X", "X"),
            ("Y", "Y"),
            ("Z", "Z"),
            ("T", "T"),
            ("x", "X"),  # case-insensitive
        ],
    )
    def test_axis_attribute(self, axis_val, expected):
        coord = xr.DataArray([1, 2], dims=["foo"], attrs={"axis": axis_val})
        assert _detect_axis_for_coord("foo", coord) == expected

    def test_axis_attribute_takes_priority(self):
        """axis attribute should override standard_name and name heuristic."""
        coord = xr.DataArray(
            [1, 2],
            dims=["lon"],
            attrs={"axis": "Y", "standard_name": "longitude"},
        )
        # axis=Y should win over standard_name=longitude and name=lon
        assert _detect_axis_for_coord("lon", coord) == "Y"


# ---------------------------------------------------------------------------
# _detect_axis_for_coord — Priority 2: standard_name attribute
# ---------------------------------------------------------------------------


class TestDetectByStandardName:
    """Test detection via the ``standard_name`` coordinate attribute."""

    @pytest.mark.parametrize(
        "standard_name,expected",
        [
            ("longitude", "X"),
            ("grid_longitude", "X"),
            ("projection_x_coordinate", "X"),
            ("latitude", "Y"),
            ("grid_latitude", "Y"),
            ("projection_y_coordinate", "Y"),
            ("altitude", "Z"),
            ("height", "Z"),
            ("depth", "Z"),
            ("air_pressure", "Z"),
            ("geopotential_height", "Z"),
            ("ocean_sigma_coordinate", "Z"),
            ("time", "T"),
            ("forecast_reference_time", "T"),
        ],
    )
    def test_standard_name(self, standard_name, expected):
        coord = xr.DataArray([1, 2], dims=["foo"], attrs={"standard_name": standard_name})
        assert _detect_axis_for_coord("foo", coord) == expected

    def test_standard_name_takes_priority_over_units(self):
        """standard_name should override units heuristic."""
        coord = xr.DataArray(
            [1, 2],
            dims=["foo"],
            attrs={"standard_name": "latitude", "units": "degrees_east"},
        )
        assert _detect_axis_for_coord("foo", coord) == "Y"


# ---------------------------------------------------------------------------
# _detect_axis_for_coord — Priority 3: units attribute
# ---------------------------------------------------------------------------


class TestDetectByUnits:
    """Test detection via the ``units`` coordinate attribute."""

    @pytest.mark.parametrize(
        "units,expected",
        [
            ("degrees_east", "X"),
            ("degree_east", "X"),
            ("degrees_E", "X"),
            ("degree_E", "X"),
            ("degreesEast", "X"),
            ("degrees_north", "Y"),
            ("degree_north", "Y"),
            ("degrees_N", "Y"),
            ("Pa", "Z"),
            ("hPa", "Z"),
            ("mbar", "Z"),
            ("millibar", "Z"),
            ("bar", "Z"),
            ("atm", "Z"),
        ],
    )
    def test_units(self, units, expected):
        coord = xr.DataArray([1, 2], dims=["foo"], attrs={"units": units})
        assert _detect_axis_for_coord("foo", coord) == expected


# ---------------------------------------------------------------------------
# _detect_axis_for_coord — Priority 4: name heuristic
# ---------------------------------------------------------------------------


class TestDetectByNameHeuristic:
    """Test detection via variable name patterns."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("lon", "X"),
            ("longitude", "X"),
            ("LON", "X"),
            ("x", "X"),
            ("X", "X"),
            ("lat", "Y"),
            ("latitude", "Y"),
            ("LAT", "Y"),
            ("y", "Y"),
            ("Y", "Y"),
            ("z", "Z"),
            ("lev", "Z"),
            ("level", "Z"),
            ("depth", "Z"),
            ("altitude", "Z"),
            ("height", "Z"),
            ("isobaric", "Z"),
            ("pressure", "Z"),
            ("sigma", "Z"),
            ("s_rho", "Z"),
            ("zlev", "Z"),
            ("time", "T"),
            ("TIME", "T"),
        ],
    )
    def test_name_pattern(self, name, expected):
        coord = xr.DataArray([1, 2], dims=["d0"])
        assert _detect_axis_for_coord(name, coord) == expected

    def test_unknown_name_returns_none(self):
        coord = xr.DataArray([1, 2], dims=["d0"])
        assert _detect_axis_for_coord("salinity", coord) is None

    def test_partial_match_not_detected(self):
        """Names that partially match patterns should not be detected."""
        coord = xr.DataArray([1, 2], dims=["d0"])
        # 'longevity' should NOT match 'lon' or 'longitude'
        assert _detect_axis_for_coord("longevity", coord) is None
        # 'lateral' should NOT match 'lat' or 'latitude'
        assert _detect_axis_for_coord("lateral", coord) is None


# ---------------------------------------------------------------------------
# detect_axes — integration tests
# ---------------------------------------------------------------------------


class TestDetectAxes:
    """Test the top-level detect_axes function."""

    def test_basic_detection(self):
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["lat", "lon"],
            coords={"lat": np.arange(3), "lon": np.arange(4)},
        )
        axes = detect_axes(da)
        assert axes["X"] == "lon"
        assert axes["Y"] == "lat"

    def test_cf_attributes_full(self):
        """Detection from axis attributes on all four axes."""
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

    def test_first_match_wins(self):
        """If two coords map to the same axis, the first one wins."""
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["lat", "latitude"],
            coords={
                "lat": np.arange(3),
                "latitude": np.arange(4),
            },
        )
        axes = detect_axes(da)
        assert axes["Y"] in ("lat", "latitude")

    def test_mixed_detection_strategies(self):
        """Mix of axis attr, standard_name, and name heuristic."""
        da = xr.DataArray(
            np.zeros((2, 3, 4)),
            dims=["level", "y_dim", "x_dim"],
            coords={
                "x_coord": ("x_dim", np.arange(4), {"standard_name": "longitude"}),
                "y_coord": ("y_dim", np.arange(3), {"units": "degrees_north"}),
                "level": np.arange(2),  # name heuristic
            },
        )
        axes = detect_axes(da)
        assert axes["X"] == "x_coord"
        assert axes["Y"] == "y_coord"
        assert axes["Z"] == "level"

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
        """Scalar (0-dim) coordinates should be ignored by detect_axes.

        After .sel(level=500), 'level' becomes a 0-dim coord and should
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


# ---------------------------------------------------------------------------
# PyVistaAccessor auto-detection integration
# ---------------------------------------------------------------------------


class TestAccessorAutoDetect:
    """Test that accessor.mesh() auto-detects coordinates."""

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
