from io import open as io_open
import os

from setuptools import setup

dirname = os.path.dirname(__file__)
readme_file = os.path.join(dirname, "README.md")
if os.path.exists(readme_file):
    with io_open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    # When this is first installed in development Docker, README.md is not available
    long_description = ""

# major, minor, patch
version_info = 0, 1, 2
# Nice string for the version
__version__ = ".".join(map(str, version_info))

setup(
    name="pyvista-xarray",
    version=__version__,
    description="xarray DataArray accessors for PyVista",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kitware, Inc.",
    author_email="kitware@kitware.com",
    url="https://github.com/pyvista/pyvista-xarray",
    packages=["pvxarray"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
    ],
    python_requires=">=3.8",
    install_requires=[
        "xarray<2022.9.0,>2022.3.0",
        "pyvista",  # TODO: set minimum version for shared array patch in https://github.com/pyvista/pyvista/pull/2697
        "scooby",
    ],
    entry_points={
        "xarray.backends": ["pyvista=pvxarray.io:PyVistaBackendEntrypoint"],
    },
)
