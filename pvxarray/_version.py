from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyvista-xarray")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = None
