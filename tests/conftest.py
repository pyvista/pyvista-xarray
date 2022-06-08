import numpy as np
import pytest

import pvxarray  # noqa: F401


@pytest.fixture
def meshgrid():
    x = np.arange(-10, 10, 0.25)
    y = np.arange(-10, 10, 0.25)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))
    return x, y, z
