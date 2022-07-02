from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import xarray as xr


@pytest.fixture
def vtr_path():
    return Path(Path(__file__).parent, "data", "air_temperature.vtr").absolute()


@pytest.fixture
def vts_path():
    return Path(Path(__file__).parent, "data", "structured.vts").absolute()


@pytest.fixture
def vti_path():
    return Path(Path(__file__).parent, "data", "wavelet.vti").absolute()


def test_read_vtr(vtr_path):
    ds = xr.open_dataset(vtr_path, engine="pyvista")
    truth = pv.RectilinearGrid(vtr_path)
    assert np.allclose(ds["air"].values.ravel(), truth["air"].ravel())
    assert np.allclose(ds["x"].values, truth.x)
    assert np.allclose(ds["y"].values, truth.y)
    assert np.allclose(ds["z"].values, truth.z)
    assert ds["air"].pyvista.mesh(x="x", y="y", z="z") == truth


def test_read_vti(vti_path):
    ds = xr.open_dataset(vti_path, engine="pyvista")
    truth = pv.UniformGrid(vti_path).cast_to_rectilinear_grid()
    assert np.allclose(ds["RTData"].values.ravel(), truth["RTData"].ravel())
    assert np.allclose(ds["x"].values, truth.x)
    assert np.allclose(ds["y"].values, truth.y)
    assert np.allclose(ds["z"].values, truth.z)
    assert ds["RTData"].pyvista.mesh(x="x", y="y", z="z") == truth


def test_read_vts(vts_path):
    ds = xr.open_dataset(vts_path, engine="pyvista")
    truth = pv.StructuredGrid(vts_path)
    assert np.allclose(ds["Elevation"].values.ravel(), truth["Elevation"].ravel())
    assert np.allclose(ds["x"].values, truth.x)
    assert np.allclose(ds["y"].values, truth.y)
    assert np.allclose(ds["z"].values, truth.z)
    assert ds["Elevation"].pyvista.mesh(x="x", y="y", z="z") == truth
