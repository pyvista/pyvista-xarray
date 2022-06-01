import numpy as np
import pytest
import xarray as xr

import pvxarray


@pytest.fixture
def bad_coord_names():
    temp = 15 + 8 * np.random.randn(2, 2, 2, 2)
    return xr.Dataset(
        {
            "temperature": (["w", "u", "v", "t"], temp),
        },
        coords={
            "ux": (["u"], np.array([-99.83, -99.32])),
            "uy": (["v"], np.array([42.25, 42.21])),
            "uz": (["w"], np.array([0, 10])),
            "t": (["t"], np.array([0.5, 1.5])),
        },
    ).temperature


@pytest.fixture
def good_coord_names(bad_coord_names):
    bad_coord_names.pyvista.x_coord = "ux"
    bad_coord_names.pyvista.y_coord = "uy"
    bad_coord_names.pyvista.z_coord = "uz"
    return bad_coord_names


def test_indexing(good_coord_names):
    ds = good_coord_names.pyvista[dict(t=0)]
    assert ds.t == 0.5
    ds = good_coord_names.pyvista[dict(t=1)]
    assert ds.t == 1.5
    ds = good_coord_names.pyvista.loc[dict(t=0.5)]
    assert ds.t == 0.5
    ds = good_coord_names.pyvista.loc[dict(t=1.5)]
    assert ds.t == 1.5


def test_set_bad_coord(bad_coord_names):
    ds = bad_coord_names[dict(t=0)]
    with pytest.raises(KeyError):
        ds.pyvista.x_coord = "foo"
    with pytest.raises(KeyError):
        ds.pyvista.y_coord = "foo"
    with pytest.raises(KeyError):
        ds.pyvista.z_coord = "foo"


def test_bad_coord(bad_coord_names):
    ds = bad_coord_names[dict(t=0)]
    assert ds.pyvista.x_coord is None
    assert ds.pyvista.y_coord is None
    assert ds.pyvista.z_coord is None
    with pytest.raises(ValueError):
        ds.pyvista.x
    with pytest.raises(ValueError):
        ds.pyvista.y
    with pytest.raises(ValueError):
        ds.pyvista.mesh


def test_report():
    assert pvxarray.Report()
