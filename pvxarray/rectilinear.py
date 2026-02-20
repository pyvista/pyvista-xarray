"""Create PyVista RectilinearGrid meshes from xarray DataArrays.

RectilinearGrid is the most memory-efficient mesh type: it stores 1D
coordinate arrays that define axis-aligned grid lines, and VTK
reconstructs the full 3D grid implicitly. This preserves zero-copy
memory sharing between xarray and VTK for both coordinates and data.

Use this when coordinates are 1D (one value per grid line), which
is the case for most regular lat/lon/level grids.
"""

from __future__ import annotations

import warnings

import pyvista as pv

from pvxarray.errors import DataCopyWarning


def mesh(
    self,
    x: str | None = None,
    y: str | None = None,
    z: str | None = None,
    order: str | None = "C",
    component: str | None = None,
    scales: dict | None = None,
):
    """Create a :class:`pyvista.RectilinearGrid` from 1D coordinates.

    Parameters
    ----------
    self : PyVistaAccessor
        The accessor instance (passed internally).
    x, y, z : str, optional
        Names of the coordinate variables for each axis.
        At least one must be specified.
    order : str, default "C"
        Array memory layout for flattening data values.
    component : str, optional
        Name of an extra dimension for multi-component arrays
        (e.g. ``"band"`` for RGB data). Triggers a data copy.
    scales : dict, optional
        Scale factors for non-numeric coordinates (replaced by
        indices that are multiplied by the scale value).

    Returns
    -------
    pyvista.RectilinearGrid
        The mesh with data values as point data. Coordinates and
        data share memory with the source xarray DataArray when
        possible (no copies for numeric, C-contiguous data).

    Notes
    -----
    **Point data vs cell data.** A ``RectilinearGrid`` with ``N``
    coordinate values along an axis has ``N`` points but only
    ``N - 1`` cells. This method assigns the DataArray values as
    **point data** (one value per grid node). VTK interpolates
    point data smoothly across cell faces when rendering.

    This works naturally when xarray coordinates are cell centers
    (the common convention), because the number of coordinate
    values equals the number of data values.

    If your coordinates represent cell **boundaries** (``N + 1``
    edges for ``N`` data values), the shape will not match and
    you will get a ``ValueError``. In that case, build the mesh
    manually with cell data::

        import pyvista as pv

        grid = pv.RectilinearGrid(lon_edges, lat_edges)
        grid.cell_data["temperature"] = da.values.ravel()

    To convert an existing point-data mesh to cell data for
    flat-shaded rendering::

        mesh = mesh.point_data_to_cell_data()

    This averages neighboring node values and is not lossless.
    """
    if order is None:
        order = "C"
    self._mesh = pv.RectilinearGrid()
    ndim = 3 - (x, y, z).count(None)
    if ndim < 1:
        raise ValueError("You must specify at least one dimension as X, Y, or Z.")
    # Construct the mesh
    if x is not None:
        self._mesh.x = self._get_array(x, scale=(scales and scales.get(x)) or 1)
    if y is not None:
        self._mesh.y = self._get_array(y, scale=(scales and scales.get(y)) or 1)
    if z is not None:
        self._mesh.z = self._get_array(z, scale=(scales and scales.get(z)) or 1)
    # Handle data values
    values = self.data
    values_dim = values.ndim
    if component is not None:
        # Assuming additional component array
        dims = [d for d in self._obj.dims if d != component]
        values = self._obj.transpose(*dims, component, transpose_coords=True).values
        values = values.reshape((-1, values.shape[-1]), order=order)
        warnings.warn(
            DataCopyWarning(
                "Made a copy of the multicomponent array - VTK/PyVista data not shared with xarray."
            ),
            stacklevel=2,
        )
        ndim += 1
    else:
        values = values.ravel(order=order)
    # Check dimensionality of data
    if values_dim != ndim:
        msg = (
            f"Dimensional mismatch between specified X, Y, Z coords "
            f"and dimensionality of DataArray ({ndim} vs {values_dim})"
        )
        if ndim > values_dim:
            raise ValueError(
                f"{msg}. Too many coordinate dimensions specified leave out Y and/or Z."
            )
        raise ValueError(
            f"{msg}. Too few coordinate dimensions specified. Be sure to specify "
            f"Y and/or Z or reduce the dimensionality of the DataArray by indexing "
            f"along non-spatial coordinates like Time."
        )
    self._mesh[self._obj.name or "data"] = values
    return self._mesh
