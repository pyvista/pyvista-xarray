from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import scooby
import xarray as xr

from pvxarray import pyvista_to_xarray


@pytest.fixture
def vtr_path():
    return Path(Path(__file__).parent, "data", "air_temperature.vtr").absolute()


@pytest.fixture
def vts_path():
    return Path(Path(__file__).parent, "data", "structured.vts").absolute()


@pytest.fixture
def vti_path():
    return Path(Path(__file__).parent, "data", "wavelet.vti").absolute()


def test_engine_is_available():
    assert "pyvista" in xr.backends.list_engines()


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


def test_convert_vtr(vtr_path):
    truth = pv.RectilinearGrid(vtr_path)
    ds = pyvista_to_xarray(truth)
    mesh = ds["air"].pyvista.mesh(x="x", y="y", z="z")
    assert np.array_equal(ds["air"].values.ravel(), truth["air"].ravel())
    assert np.may_share_memory(ds["air"].values.ravel(), truth["air"].ravel())
    assert np.array_equal(mesh.x, truth.x)
    assert np.array_equal(mesh.y, truth.y)
    assert np.array_equal(mesh.z, truth.z)
    assert np.may_share_memory(mesh.z, truth.z)
    assert mesh == truth

    # TODO: figure out why this is failing
    #   broke after https://github.com/pyvista/pyvista/pull/3179
    if not scooby.meets_version(pv.__version__, "0.37"):
        assert np.may_share_memory(mesh.x, truth.x)
        assert np.may_share_memory(mesh.y, truth.y)


def test_convert_vti(vti_path):
    truth = pv.UniformGrid(vti_path)
    truth_r = truth.cast_to_rectilinear_grid()
    ds = pyvista_to_xarray(truth)
    mesh = ds["RTData"].pyvista.mesh(x="x", y="y", z="z")
    assert np.array_equal(ds["RTData"].values.ravel(), truth["RTData"].ravel())
    assert np.may_share_memory(ds["RTData"].values.ravel(), truth["RTData"].ravel())
    assert np.array_equal(mesh.x, truth_r.x)
    assert np.array_equal(mesh.y, truth_r.y)
    assert np.array_equal(mesh.z, truth_r.z)
    assert mesh == truth_r


def test_convert_vts(vts_path):
    truth = pv.StructuredGrid(vts_path)
    ds = pyvista_to_xarray(truth)
    assert np.array_equal(ds["Elevation"].values.ravel(), truth["Elevation"].ravel())
    assert np.may_share_memory(ds["Elevation"].values.ravel(), truth["Elevation"].ravel())
    mesh = ds["Elevation"].pyvista.mesh(x="x", y="y", z="z")
    assert np.array_equal(mesh.x, truth.x)
    assert np.array_equal(mesh.y, truth.y)
    assert np.array_equal(mesh.z, truth.z)
    assert mesh == truth
