"""Custom warning classes for pvxarray."""


class DataCopyWarning(Warning):
    """Issued when data must be copied instead of shared with VTK.

    Some mesh types (e.g. :class:`pyvista.StructuredGrid`) and operations
    (e.g. multi-component array handling) require rearranging data in
    memory, which breaks the zero-copy sharing between xarray and VTK.
    """

    pass
