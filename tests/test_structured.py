import numpy as np
import pytest
import xarray as xr

import pvxarray  # noqa: F401


@pytest.fixture
def simple():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    z = np.sin(np.sqrt(x**2 + y**2))
    x, y, z = np.meshgrid(x, y, z)
    temp = 15 + 8 * np.random.randn(*x.shape)

    ds = xr.Dataset(
        {
            "temperature": (["zi", "xi", "yi"], temp),
        },
        coords={
            "x": (["xi", "yi", "zi"], x),
            "y": (["xi", "yi", "zi"], y),
            "z": (["xi", "yi", "zi"], z),
        },
    )

    return {"x": x, "y": y, "z": z, "temp": temp, "ds": ds}


@pytest.fixture
def roms():
    ds = xr.tutorial.open_dataset("ROMS_example.nc", chunks={"ocean_time": 1})

    if ds.Vtransform == 1:
        Zo_rho = ds.hc * (ds.s_rho - ds.Cs_r) + ds.Cs_r * ds.h
        z_rho = Zo_rho + ds.zeta * (1 + Zo_rho / ds.h)
    elif ds.Vtransform == 2:
        Zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
        z_rho = ds.zeta + (ds.zeta + ds.h) * Zo_rho

    ds.coords["z_rho"] = z_rho.transpose()  # needing transpose seems to be an xarray bug

    return ds


def test_simple(simple):
    mesh = simple["ds"].temperature.pyvista.mesh(x="x", y="y", z="z")

    assert mesh.n_points == simple["x"].size
    assert np.array_equal(mesh.x, simple["x"])
    assert np.array_equal(mesh.y, simple["y"])
    assert np.array_equal(mesh.z, simple["z"])
    assert np.array_equal(mesh["temperature"], simple["temp"].ravel(order="F"))


@pytest.mark.xfail
def test_shared_data(simple):
    mesh = simple["ds"].temperature.pyvista.mesh

    mesh["temperature"][0] = -1
    assert simple["temp"].ravel()[0] == -1
    assert np.array_equal(mesh["temperature"], simple["temp"].ravel(order="F"))


def test_roms(roms):
    ds = roms
    da = ds.salt[{"ocean_time": 0}]

    # Make array ordering consistent
    da = da.transpose("s_rho", "xi_rho", "eta_rho", transpose_coords=False)

    # Grab StructuredGrid mesh
    mesh = da.pyvista.mesh(x="lon_rho", y="lat_rho", z="z_rho")

    assert np.allclose(mesh["salt"], da.values.ravel(order="F"), equal_nan=True)
    assert np.allclose(mesh.z, da.z_rho, equal_nan=True)


def test_component_not_supported(simple):
    with pytest.raises(ValueError, match="not currently supported"):
        simple["ds"].temperature.pyvista.mesh(x="x", y="y", z="z", component="zi")


def test_1d_structured_error():
    x = np.arange(10, dtype=float)
    temp = np.random.randn(10)
    da = xr.DataArray(
        temp,
        dims=["xi"],
        coords={"x": (["xi"], x)},
        name="temp",
    )
    with pytest.raises(ValueError, match="rectilinear"):
        da.pyvista.mesh(x="x", mesh_type="structured")


def test_structured_scales():
    x_vals = np.arange(3, dtype=float)
    x, t = np.meshgrid(x_vals, np.arange(3))
    temp = np.random.randn(*x.shape)
    da = xr.DataArray(
        temp,
        dims=["ti", "xi"],
        coords={
            "x": (["ti", "xi"], x.astype(float)),
            "time": (["ti", "xi"], t.astype(float)),
        },
        name="temp",
    )
    # Scales only affect non-numeric coords, but for structured grids
    # with 2D numeric coords, they pass through _get_array unchanged.
    # Just verify scales kwarg is accepted without error.
    mesh = da.pyvista.mesh(x="x", y="time", mesh_type="structured", scales={"x": 2.0})
    assert mesh.n_points == 9


def test_structured_no_data_name():
    x = np.arange(-2, 2, 1.0)
    y = np.arange(-2, 2, 1.0)
    x, y = np.meshgrid(x, y)
    temp = np.random.randn(*x.shape)
    da = xr.DataArray(
        temp,
        dims=["xi", "yi"],
        coords={"x": (["xi", "yi"], x), "y": (["xi", "yi"], y)},
    )
    mesh = da.pyvista.mesh(x="x", y="y", mesh_type="structured")
    assert "data" in mesh.point_data
