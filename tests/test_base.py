import numpy as np
import pytest
import xarray as xr

import pvxarray


@pytest.fixture
def sample():
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


def test_accessor_available():
    da = xr.DataArray()
    assert hasattr(da, "pyvista")


def test_indexing(sample):
    ds = sample.pyvista[dict(t=0)]
    assert ds.t == 0.5
    ds = sample.pyvista[dict(t=1)]
    assert ds.t == 1.5
    ds = sample.pyvista.loc[dict(t=0.5)]
    assert ds.t == 0.5
    ds = sample.pyvista.loc[dict(t=1.5)]
    assert ds.t == 1.5


def test_report():
    assert pvxarray.Report()


def test_bad_key(sample):
    with pytest.raises(KeyError):
        sample[dict(t=0)].pyvista.mesh(x="foo")
    with pytest.raises(KeyError):
        sample[dict(t=0)].pyvista.mesh(x="ux", y="hello")
    mesh = sample[dict(t=0)].pyvista.mesh(x="ux", y="uy", z="uz")
    assert mesh.n_points


def test_forgot_choose_time(sample):
    with pytest.raises(ValueError):
        sample.pyvista.mesh(x="ux", y="uy", z="uz")
