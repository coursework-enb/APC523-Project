from .core import (
    NavierStokesSolver2D,
    SpatialDiscretizationStrategy,
    TimeIntegratorStrategy,
)
from .spatial_discretization import (
    FiniteDifferenceDiscretizer,
    FiniteDifferenceUpwindDiscretizer,
    FiniteVolumeDiscretizer,
)

__all__ = [
    "FiniteDifferenceDiscretizer",
    "FiniteDifferenceUpwindDiscretizer",
    "FiniteVolumeDiscretizer",
    "NavierStokesSolver2D",
    "SpatialDiscretizationStrategy",
    "TimeIntegratorStrategy",
]
__version__ = "0.1.0"
