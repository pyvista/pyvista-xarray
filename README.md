# PyVista xarray

[![PyPI](https://img.shields.io/pypi/v/pyvista-xarray.svg?logo=python&logoColor=white)](https://pypi.org/project/pyvista-xarray/)
[![codecov](https://codecov.io/gh/pyvista/pyvista-xarray/branch/main/graph/badge.svg?token=4BSDVV0WOG)](https://codecov.io/gh/pyvista/pyvista-xarray)
[![MyBinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyvista/pyvista-xarray/HEAD)

xarray DataArray accessors for PyVista to visualize datasets in 3D

## Usage

Import `pvxarray` to register the `.pyvista` accessor on xarray `DataArray`
objects. This gives you access to methods for creating 3D meshes, plotting,
and lazy evaluation of large datasets.

Try on MyBinder: https://mybinder.org/v2/gh/pyvista/pyvista-xarray/HEAD

```py
import pvxarray
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air[dict(time=0)]

# Plot in 3D
da.pyvista.plot(x="lon", y="lat", show_edges=True, cpos='xy')

# Or grab the mesh object for use with PyVista
mesh = da.pyvista.mesh(x="lon", y="lat")
```

<!-- notebook=0, off_screen=1, screenshot='imgs/air_temperature.png' -->

![air_temperature](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/air_temperature.png)

### Coordinate Auto-Detection

If your data follows [CF conventions](https://cfconventions.org/), you can
omit the `x`, `y`, and `z` arguments entirely. `pyvista-xarray` uses
[cf-xarray](https://cf-xarray.readthedocs.io/) to detect coordinate axes
from attributes like `axis`, `standard_name`, and `units`, as well as
variable name heuristics:

```py
import pvxarray
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air[dict(time=0)]

# Coordinates are auto-detected from CF attributes
mesh = da.pyvista.mesh()

# Inspect the detected axes
da.pyvista.axes
# {'X': 'lon', 'Y': 'lat'}
```

### Lazy Evaluation with Algorithm Sources

For large or dask-backed datasets, create a VTK algorithm source that lazily
evaluates data on demand. This avoids loading the entire dataset into memory
and supports time stepping, resolution control, and spatial slicing:

```py
import pvxarray
import pyvista as pv
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air

# Create a lazy algorithm source with time stepping
source = da.pyvista.algorithm(x="lon", y="lat", time="time")

# Add directly to a plotter
pl = pv.Plotter()
pl.add_mesh(source)
pl.show(cpos="xy")

# Step through time
source.time_index = 10
```

Use the `resolution` parameter to downsample large datasets for interactive
rendering:

```py
source = da.pyvista.algorithm(x="lon", y="lat", time="time", resolution=0.5)
```

### Reading VTK Files as xarray Datasets

Read VTK mesh files directly into xarray using the `pyvista` backend
engine. Supported formats include `.vti`, `.vtr`, `.vts`, and `.vtk`:

```py
import xarray as xr

ds = xr.open_dataset("data.vtk", engine="pyvista")
ds["data array"].pyvista.plot(x="x", y="y", z="z")
```

### Converting PyVista Meshes to xarray

Convert PyVista meshes back to xarray Datasets with `pyvista_to_xarray`.
Supported mesh types: `RectilinearGrid`, `ImageData`, and `StructuredGrid`:

```py
import pyvista as pv
from pvxarray import pyvista_to_xarray

grid = pv.RectilinearGrid([0, 1, 2], [0, 1], [0, 1])
grid["values"] = range(grid.n_points)
ds = pyvista_to_xarray(grid)
```

## Installation

```bash
pip install 'pyvista-xarray[jupyter]'
```

This includes Jupyter rendering support (via Trame), common I/O libraries
(`netcdf4`, `rioxarray`), and dask for lazy evaluation. For a minimal
install without these extras:

```bash
pip install pyvista-xarray
```

`pyvista-xarray` is also available on conda-forge:

```bash
conda install -c conda-forge pyvista-xarray
```

## Examples

The [`examples/`](https://github.com/pyvista/pyvista-xarray/tree/main/examples)
directory contains Jupyter notebooks demonstrating various use cases:

| Notebook                                                      | Description                                              |
| ------------------------------------------------------------- | -------------------------------------------------------- |
| [introduction.ipynb](examples/introduction.ipynb)             | Quick start with auto-detection, rioxarray, and 3D grids |
| [simple.ipynb](examples/simple.ipynb)                         | Lazy evaluation, time stepping, and algorithm sources    |
| [ocean_model.ipynb](examples/ocean_model.ipynb)               | Curvilinear grids with ROMS ocean model data             |
| [atmospheric_levels.ipynb](examples/atmospheric_levels.ipynb) | 3D atmospheric data across pressure levels               |
| [lightning.ipynb](examples/lightning.ipynb)                   | Point cloud visualization from scattered observations    |
| [cartographic.ipynb](examples/cartographic.ipynb)             | Geographic projections with GeoVista                     |
| [radar.ipynb](examples/radar.ipynb)                           | Radar data with polar coordinates via xradar             |
| [sea_temps.ipynb](examples/sea_temps.ipynb)                   | Sea surface temperature raster data                      |

There are also Python scripts for interactive Trame web applications:
`examples/level_of_detail.py` and `examples/level_of_detail_geovista.py`.

### Simple RectilinearGrid

```py
import numpy as np
import pvxarray
import xarray as xr

lon = np.array([-99.83, -99.32])
lat = np.array([42.25, 42.21])
z = np.array([0, 10])
temp = 15 + 8 * np.random.randn(2, 2, 2)

ds = xr.Dataset(
    {
        "temperature": (["z", "x", "y"], temp),
    },
    coords={
        "lon": (["x"], lon),
        "lat": (["y"], lat),
        "z": (["z"], z),
    },
)

mesh = ds.temperature.pyvista.mesh(x="lon", y="lat", z="z")
mesh.plot()
```

### Raster with rioxarray

```py
import pvxarray
import rioxarray
import xarray as xr

da = rioxarray.open_rasterio("TC_NG_SFBay_US_Geo_COG.tif")
da = da.rio.reproject("EPSG:3857")

# Grab the mesh object for use with PyVista
mesh = da.pyvista.mesh(x="x", y="y", component="band")

mesh.plot(scalars="data", cpos='xy', rgb=True)
```

<!-- notebook=0, off_screen=1, screenshot='imgs/raster.png' -->

![raster](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/raster.png)

```py
import pvxarray
import rioxarray

da = rioxarray.open_rasterio("Elevation.tif")
da = da.rio.reproject("EPSG:3857")

# Grab the mesh object for use with PyVista
mesh = da.pyvista.mesh(x="x", y="y")

# Warp top and plot in 3D
mesh.warp_by_scalar().plot()
```

<!-- notebook=0, off_screen=1, screenshot='imgs/topo.png' -->

![topo](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/topo.png)

### StructuredGrid

```py
import pvxarray
import pyvista as pv
import xarray as xr

ds = xr.tutorial.open_dataset("ROMS_example.nc", chunks={"ocean_time": 1})

if ds.Vtransform == 1:
    Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
    z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / ds.h)
elif ds.Vtransform == 2:
    Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho

ds.coords["z_rho"] = z_rho.transpose()  # needing transpose seems to be an xarray bug

da = ds.salt[dict(ocean_time=0)]

# Make array ordering consistent
da = da.transpose("s_rho", "xi_rho", "eta_rho", transpose_coords=False)

# Grab StructuredGrid mesh
mesh = da.pyvista.mesh(x="lon_rho", y="lat_rho", z="z_rho")

# Plot in 3D
p = pv.Plotter()
p.add_mesh(mesh, lighting=False, cmap='plasma', clim=[0, 35])
p.view_vector([1, -1, 1])
p.set_scale(zscale=0.001)
p.show()
```

![raster](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/structured.png)

## Feedback

Please share your thoughts and questions on the
[Discussions](https://github.com/pyvista/pyvista-xarray/discussions) board.
If you would like to report any bugs or make feature requests, please open an
[issue](https://github.com/pyvista/pyvista-xarray/issues).

If filing a bug report, please share a scooby Report:

```py
import pvxarray
print(pvxarray.Report())
```
