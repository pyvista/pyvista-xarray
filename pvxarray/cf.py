"""CF-convention coordinate auto-detection for xarray DataArrays.

Thin wrapper around `cf-xarray <https://cf-xarray.readthedocs.io/>`_
for detecting spatial (X, Y, Z) and temporal (T) coordinate axes.
"""

from __future__ import annotations

import cf_xarray  # noqa: F401 — registers .cf accessor
import xarray as xr

# cf-xarray coordinate types mapped to axis labels.
# cf-xarray separates "axes" (from explicit axis attributes) and
# "coordinates" (from standard_name, units, name heuristics). We
# combine both to build a complete axis mapping.
_COORD_TYPE_TO_AXIS: dict[str, str] = {
    "longitude": "X",
    "latitude": "Y",
    "vertical": "Z",
    "time": "T",
}


def detect_axes(da: xr.DataArray) -> dict[str, str]:
    """Detect CF axis mapping from a DataArray's coordinates.

    Uses cf-xarray to inspect coordinate attributes (``axis``,
    ``standard_name``, ``units``) and variable name heuristics to
    map coordinates to axis types (``"X"``, ``"Y"``, ``"Z"``, ``"T"``).

    Scalar (0-dimensional) coordinates are excluded since they cannot
    represent a spatial axis of the data (e.g. after ``.sel(level=500)``).

    Parameters
    ----------
    da : xr.DataArray
        The data array whose coordinates to inspect.

    Returns
    -------
    dict[str, str]
        Mapping from axis type to coordinate name, e.g.
        ``{"X": "lon", "Y": "lat", "T": "time"}``.
        Only detected axes are included; first match per axis wins.
    """
    # guess_coord_axis() adds axis/standard_name attrs to coordinates
    # that lack them, using cf-xarray's heuristics (name patterns,
    # datetime dtype detection, etc.).
    guessed = da.cf.guess_coord_axis()

    axes: dict[str, str] = {}

    def _try_add(axis: str, name: str) -> None:
        """Add name to axes[axis] if not already set and non-scalar."""
        if axis not in axes and name in da.coords and da.coords[name].ndim > 0:
            axes[axis] = name

    # 1. Explicit axes from cf-xarray (axis attribute, _CoordinateAxisType)
    for axis, names in guessed.cf.axes.items():
        for name in names:
            _try_add(axis, name)

    # 2. Coordinate-type detection (standard_name, units, name heuristics)
    #    Maps "longitude"→X, "latitude"→Y, "vertical"→Z, "time"→T
    for coord_type, axis in _COORD_TYPE_TO_AXIS.items():
        for name in guessed.cf.coordinates.get(coord_type, []):
            _try_add(axis, name)

    return axes
