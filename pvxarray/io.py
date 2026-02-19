"""Read and write PyVista meshes as xarray Datasets.

Provides conversion functions from PyVista mesh types to
:class:`xarray.Dataset` objects, and a backend engine so that
``xr.open_dataset("file.vtk", engine="pyvista")`` works directly.

Supported mesh types
--------------------
- :class:`pyvista.RectilinearGrid` — axis-aligned grids with 1D
  coordinate arrays
- :class:`pyvista.ImageData` — uniform-spacing grids (VTK image data)
- :class:`pyvista.StructuredGrid` — curvilinear grids with 3D
  coordinate arrays (requires data copy)

Examples
--------
>>> import xarray as xr
>>> ds = xr.open_dataset("data.vtr", engine="pyvista")
"""

from __future__ import annotations

import os
from typing import ClassVar
import warnings

import numpy as np
import pyvista as pv
from pyvista import ImageData
import xarray as xr
from xarray.backends import BackendEntrypoint

from pvxarray.errors import DataCopyWarning


def rectilinear_grid_to_dataset(mesh: pv.RectilinearGrid) -> xr.Dataset:
    """Convert a :class:`pyvista.RectilinearGrid` to an xarray Dataset.

    Parameters
    ----------
    mesh : pyvista.RectilinearGrid
        The rectilinear grid mesh to convert.

    Returns
    -------
    xarray.Dataset
        Dataset with data variables from point data and 1D coordinate
        arrays for each axis.
    """
    dims = list(mesh.dimensions)
    dims = dims[-1:] + dims[:-1]
    return xr.Dataset(
        {
            name: (["z", "x", "y"], arr.ravel().reshape(dims))
            for name, arr in mesh.point_data.items()
        },
        coords={
            "x": (["x"], mesh.x),
            "y": (["y"], mesh.y),
            "z": (["z"], mesh.z),
        },
    )


def image_data_to_dataset(mesh: ImageData) -> xr.Dataset:
    """Convert a :class:`pyvista.ImageData` to an xarray Dataset.

    Generates coordinate arrays from the mesh extent, spacing, and
    origin.

    Parameters
    ----------
    mesh : pyvista.ImageData
        The uniform grid (image data) mesh to convert.

    Returns
    -------
    xarray.Dataset
        Dataset with data variables from point data and generated
        coordinate arrays.
    """
    extent = mesh.GetExtent()

    def gen_coords(i):
        return np.arange(extent[2 * i], extent[2 * i + 1] + 1) * mesh.spacing[i] + mesh.origin[i]

    dims = list(mesh.dimensions)
    dims = dims[-1:] + dims[:-1]
    return xr.Dataset(
        {
            name: (["z", "x", "y"], arr.ravel().reshape(dims))
            for name, arr in mesh.point_data.items()
        },
        coords={
            "x": (["x"], gen_coords(0)),
            "y": (["y"], gen_coords(1)),
            "z": (["z"], gen_coords(2)),
        },
    )


def structured_grid_to_dataset(mesh: pv.StructuredGrid) -> xr.Dataset:
    """Convert a :class:`pyvista.StructuredGrid` to an xarray Dataset.

    Parameters
    ----------
    mesh : pyvista.StructuredGrid
        The structured grid mesh to convert.

    Returns
    -------
    xarray.Dataset
        Dataset with data variables from point data and 3D coordinate
        arrays for curvilinear coordinates.

    Warns
    -----
    DataCopyWarning
        Always emitted because structured grids store interleaved
        points that must be reshaped.
    """
    warnings.warn(
        DataCopyWarning(
            "StructuredGrid dataset engine duplicates data - VTK/PyVista data not shared with xarray."
        ),
        stacklevel=2,
    )
    return xr.Dataset(
        {
            name: (["xi", "yi", "zi"], arr.ravel().reshape(mesh.dimensions))
            for name, arr in mesh.point_data.items()
        },
        coords={
            "x": (["xi", "yi", "zi"], mesh.x),
            "y": (["xi", "yi", "zi"], mesh.y),
            "z": (["xi", "yi", "zi"], mesh.z),
        },
    )


def pyvista_to_xarray(mesh: pv.DataSet) -> xr.Dataset:
    """Convert a PyVista mesh to an xarray Dataset.

    Dispatches to the appropriate converter based on mesh type.

    Parameters
    ----------
    mesh : pyvista.DataSet
        A PyVista mesh. Supported types are
        :class:`~pyvista.RectilinearGrid`,
        :class:`~pyvista.ImageData`, and
        :class:`~pyvista.StructuredGrid`.

    Returns
    -------
    xarray.Dataset
        Dataset containing the mesh's point data as data variables
        and spatial coordinates.

    Raises
    ------
    TypeError
        If *mesh* is not a supported type.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pvxarray import pyvista_to_xarray
    >>> grid = pv.RectilinearGrid([0, 1, 2], [0, 1], [0, 1])
    >>> ds = pyvista_to_xarray(grid)
    """
    if isinstance(mesh, pv.RectilinearGrid):
        return rectilinear_grid_to_dataset(mesh)
    elif isinstance(mesh, ImageData):
        return image_data_to_dataset(mesh)
    elif isinstance(mesh, pv.StructuredGrid):
        return structured_grid_to_dataset(mesh)
    else:
        raise TypeError(
            f"pvxarray is unable to generate an xarray DataSet from the "
            f"{type(mesh).__name__} VTK/PyVista data type at this time."
        )


class PyVistaBackendEntrypoint(BackendEntrypoint):
    """Xarray backend engine for reading VTK files via PyVista.

    Enables ``xr.open_dataset("file.vtk", engine="pyvista")``.

    Supports ``.vti``, ``.vtr``, ``.vts``, and ``.vtk`` file formats.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset("mesh.vtr", engine="pyvista")
    """

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        force_ext=None,
        file_format=None,
        progress_bar=False,
    ):
        """Open a VTK file as an xarray Dataset.

        Parameters
        ----------
        filename_or_obj : str or path-like
            Path to a VTK file.
        drop_variables : sequence of str, optional
            Variable names to exclude (ignored, kept for API compat).
        force_ext : str, optional
            Override the file extension for PyVista's reader dispatch.
        file_format : str, optional
            Explicit file format string for PyVista.
        progress_bar : bool, default False
            Show a progress bar during reading.

        Returns
        -------
        xarray.Dataset
            Dataset created from the mesh's point data.
        """
        mesh = pv.read(
            filename_or_obj,
            force_ext=force_ext,
            file_format=file_format,
            progress_bar=progress_bar,
        )
        return pyvista_to_xarray(mesh)

    open_dataset_parameters: ClassVar[list[str]] = [
        "filename_or_obj",
        "attrs",
        "force_ext",
        "file_format",
        "progress_bar",
    ]

    def guess_can_open(self, filename_or_obj) -> bool:
        """Check whether a file can be opened by this backend.

        Parameters
        ----------
        filename_or_obj : str or path-like
            Path to test.

        Returns
        -------
        bool
            ``True`` if the file has a recognized VTK extension.
        """
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".vti", ".vtr", ".vts", ".vtk"}
