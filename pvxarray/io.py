import os
from typing import ClassVar
import warnings

import numpy as np
import pyvista as pv
from pyvista import ImageData
import xarray as xr
from xarray.backends import BackendEntrypoint

from pvxarray.errors import DataCopyWarning


def rectilinear_grid_to_dataset(mesh):
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


def image_data_to_dataset(mesh):
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


def structured_grid_to_dataset(mesh):
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


def pyvista_to_xarray(mesh):
    """Generate an xarray DataSet from a PyVista mesh object."""
    if isinstance(mesh, pv.RectilinearGrid):
        return rectilinear_grid_to_dataset(mesh)
    elif isinstance(mesh, ImageData):
        return image_data_to_dataset(mesh)
    elif isinstance(mesh, pv.StructuredGrid):
        return structured_grid_to_dataset(mesh)
    else:
        raise TypeError(
            f"pvxarray is unable to generate an xarray DataSet from the {type(mesh)} VTK/PyVista data type at this time."
        )


class PyVistaBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        force_ext=None,
        file_format=None,
        progress_bar=False,
    ):
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

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".vti", ".vtr", ".vts", ".vtk"}
