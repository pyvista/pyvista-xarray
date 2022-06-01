# PyVista xarray

PyVista DataArray accessor for xarray to visualize datasets in 3D


## 🚀 Usage

You must `import pvxarray` in order to register the `DataArray` accessor with
xarray. After which, a `pyvista_*` namespace of accessors will be available.

The following is an example to visualize a `RectilinearGrid` with PyVista:

```py
import pvxarray
import xarray as xr

ds = xr.tutorial.load_dataset("air_temperature")
da = ds.air[dict(time=0)]  # Select DataArray for a timestep

# Plot in 3D
da.pyvista_rectilinear.plot(show_edges=True, cpos='xy')

# Or grab the mesh object for use with PyVista
mesh = da.pyvista_rectilinear.mesh
```

<!-- notebook=0, off_screen=1, screenshot='imgs/air_temperature.png' -->
![air_temperature](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/air_temperature.png)


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

mesh = ds.temperature.pyvista_rectilinear.mesh
mesh.plot()
```


## Raster with rioxarray

```py
import pvxarray
import rioxarray
import xarray as xr

da = rioxarray.open_rasterio("TC_NG_SFBay_US_Geo_COG.tif")

# Grab a single band
band = da[dict(band=0)]

# Grab the mesh object for use with PyVista
mesh = band.pyvista_rectilinear.mesh

mesh.plot(cpos='xy')
```

![raster](https://raw.githubusercontent.com/pyvista/pyvista-xarray/main/imgs/raster.png)
