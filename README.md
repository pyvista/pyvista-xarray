# PyVista xarray

[![PyPI](https://img.shields.io/pypi/v/pyvista-xarray.svg?logo=python&logoColor=white)](https://pypi.org/project/pyvista-xarray/)
[![codecov](https://codecov.io/gh/pyvista/pyvista-xarray/branch/main/graph/badge.svg?token=4BSDVV0WOG)](https://codecov.io/gh/pyvista/pyvista-xarray)
[![MyBinder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyvista/pyvista-xarray/HEAD)

PyVista DataArray accessor for xarray to visualize datasets in 3D


## 🚀 Usage

You must `import pvxarray` in order to register the `DataArray` accessor with
xarray. After which, a `pyvista` namespace of accessors will be available.

Try on MyBinder: https://mybinder.org/v2/gh/pyvista/pyvista-xarray/HEAD

The following is an example to visualize a `RectilinearGrid` with PyVista:

```py
import pvxarray
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air[dict(time=0)]  # Select DataArray for a timestep

# Plot in 3D
da.pyvista.plot(x="lon", y="lat", show_edges=True, cpos='xy')

# Or grab the mesh object for use with PyVista
mesh = da.pyvista.mesh(x="lon", y="lat")
```

<!-- notebook=0, off_screen=1, screenshot='imgs/air_temperature.png' -->
![air_temperature](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/air_temperature.png)


Or you can read VTK meshes directly to xarray:

```py
import xarray as xr

ds = xr.open_dataset("data.vtk", engine="pyvista")
ds["data array"].pyvista.plot(x="x", y="y", z="z")
```

## ⬇️ Installation

```bash
pip install pyvista-xarray
```


## 💭 Feedback
Please share your thoughts and questions on the Discussions board. If you would
like to report any bugs or make feature requests, please open an issue.

If filing a bug report, please share a scooby Report:

```py
import pvxarray
print(pvxarray.Report())
```


## 🏏 Further Examples

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


## Raster with rioxarray

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


## StructuredGrid

```py
import pvxarray
import rioxarray
import pyvista as pv

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
