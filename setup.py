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
version_info = 0, 1, 3
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python",
    ],
    python_requires=">=3.9",
    install_requires=[
        "xarray>=2022.12.0",  # Broke between 2022.9-11
        "pyvista>=0.37",
        "scooby",
    ],
    entry_points={
        "xarray.backends": ["pyvista=pvxarray.io:PyVistaBackendEntrypoint"],
    },
)
