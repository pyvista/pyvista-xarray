"""CF-convention coordinate auto-detection for xarray DataArrays.

Detects spatial (X, Y, Z) and temporal (T) coordinate axes using a
multi-level strategy inspired by cf-xarray:

1. ``axis`` attribute (CF standard: ``"X"``, ``"Y"``, ``"Z"``, ``"T"``)
2. ``standard_name`` attribute (e.g. ``"latitude"``, ``"longitude"``)
3. ``units`` attribute (e.g. ``"degrees_east"``, ``"degrees_north"``)
4. Variable name heuristics (regex: ``lon*``, ``lat*``, ``time*``, etc.)
"""

from __future__ import annotations

import re

import xarray as xr

# --- CF standard_name → axis mapping ---
_STANDARD_NAME_TO_AXIS: dict[str, str] = {
    # X axis
    "longitude": "X",
    "grid_longitude": "X",
    "projection_x_coordinate": "X",
    # Y axis
    "latitude": "Y",
    "grid_latitude": "Y",
    "projection_y_coordinate": "Y",
    # Z axis
    "altitude": "Z",
    "height": "Z",
    "depth": "Z",
    "air_pressure": "Z",
    "geopotential_height": "Z",
    "atmosphere_ln_pressure_coordinate": "Z",
    "atmosphere_sigma_coordinate": "Z",
    "atmosphere_hybrid_sigma_pressure_coordinate": "Z",
    "ocean_sigma_coordinate": "Z",
    "ocean_s_coordinate": "Z",
    "ocean_s_coordinate_g1": "Z",
    "ocean_s_coordinate_g2": "Z",
    "ocean_double_sigma_coordinate": "Z",
    # T axis
    "time": "T",
    "forecast_reference_time": "T",
    "forecast_period": "T",
}

# --- units → axis mapping ---
_UNITS_TO_AXIS: dict[re.Pattern, str] = {
    re.compile(r"^degrees?_?(east|E)$", re.IGNORECASE): "X",
    re.compile(r"^degrees?_?(north|N)$", re.IGNORECASE): "Y",
    re.compile(r"^(Pa|hPa|mbar|millibar|bar|atm)$"): "Z",
}

# --- Variable name heuristics (lowest priority) ---
_NAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^lon(gitude)?$", re.IGNORECASE), "X"),
    (re.compile(r"^x$", re.IGNORECASE), "X"),
    (re.compile(r"^lat(itude)?$", re.IGNORECASE), "Y"),
    (re.compile(r"^y$", re.IGNORECASE), "Y"),
    (
        re.compile(
            r"^(z|lev|level|depth|altitude|height|isobaric|pressure|sigma|s_rho|zlev)$",
            re.IGNORECASE,
        ),
        "Z",
    ),
    (re.compile(r"^time$", re.IGNORECASE), "T"),
]


def _detect_axis_for_coord(name: str, coord: xr.DataArray) -> str | None:
    """Detect the CF axis type for a single coordinate variable.

    Parameters
    ----------
    name : str
        The coordinate variable name.
    coord : xr.DataArray
        The coordinate data.

    Returns
    -------
    str or None
        One of ``"X"``, ``"Y"``, ``"Z"``, ``"T"``, or ``None`` if
        no axis could be determined.
    """
    attrs = coord.attrs

    # Priority 1: explicit axis attribute
    axis = attrs.get("axis", "").upper()
    if axis in ("X", "Y", "Z", "T"):
        return axis

    # Priority 2: standard_name attribute
    standard_name = attrs.get("standard_name", "")
    if standard_name in _STANDARD_NAME_TO_AXIS:
        return _STANDARD_NAME_TO_AXIS[standard_name]

    # Priority 3: units attribute
    units = attrs.get("units", "")
    if units:
        for pattern, ax in _UNITS_TO_AXIS.items():
            if pattern.match(units):
                return ax

    # Priority 4: variable name heuristic
    for pattern, ax in _NAME_PATTERNS:
        if pattern.match(name):
            return ax

    return None


def detect_axes(da: xr.DataArray) -> dict[str, str]:
    """Detect CF axis mapping from a DataArray's coordinates.

    Inspects all coordinates and maps each to an axis type
    (``"X"``, ``"Y"``, ``"Z"``, ``"T"``) using CF conventions.
    Only the first match per axis type is retained.

    Parameters
    ----------
    da : xr.DataArray
        The data array whose coordinates to inspect.

    Returns
    -------
    dict[str, str]
        Mapping from axis type to coordinate name, e.g.
        ``{"X": "lon", "Y": "lat", "T": "time"}``.
        Only detected axes are included.
    """
    axes: dict[str, str] = {}
    # Iterate coordinates; first match per axis wins.
    # Skip scalar (0-dimensional) coordinates since they cannot
    # represent a spatial axis of the data (e.g. after .sel(level=500)).
    for name, coord in da.coords.items():
        if coord.ndim == 0:
            continue
        axis = _detect_axis_for_coord(str(name), coord)
        if axis is not None and axis not in axes:
            axes[axis] = str(name)
    return axes
