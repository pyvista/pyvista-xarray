import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from pvxarray import utils

# TODO: VTK_SILENCE_GET_VOID_POINTER_WARNINGS


def _get_xyz_bounds(values):
    return (
        values[:, 0].min(),
        values[:, 0].max(),
        values[:, 1].min(),
        values[:, 1].max(),
        values[:, 2].min(),
        values[:, 2].max(),
    )


def test_soa_values(meshgrid):
    """Generate SOA array, check size and value bounds."""
    x, y, z = meshgrid
    bounds = (x.min(), x.max(), y.min(), y.max(), z.min(), z.max())
    # Test standard SOA array
    data = utils.soa(x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F"))
    assert data.GetNumberOfValues() == x.size * 3
    dvalues = vtk_to_numpy(data)
    dbounds = _get_xyz_bounds(dvalues)
    assert dbounds == bounds
    # Test with vtkPoints
    points = vtk.vtkPoints()
    points.SetData(data)
    assert points.GetNumberOfPoints() == x.size
    pvalues = vtk_to_numpy(points.GetData())
    pbounds = _get_xyz_bounds(pvalues)
    assert pbounds == bounds


def test_soa_structured_grid(meshgrid):
    x, y, z = meshgrid
    points = utils.soa_points(x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F"))

    # Generate mesh
    mesh = vtk.vtkStructuredGrid()
    mesh.SetPoints(points)
    mesh.SetDimensions((*x.shape, 1))

    # Validate
    assert mesh.GetNumberOfPoints() == x.size
    assert mesh.GetBounds() == (x.min(), x.max(), y.min(), y.max(), z.min(), z.max())


# def test_soa_modify_inplace(meshgrid):
#     x, y, z = meshgrid
