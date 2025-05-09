# mypy: ignore-errors
# ruff: noqa

from .core import (
    NavierStokesSolver2D,
    SpatialDiscretizationStrategy,
    TimeIntegratorStrategy,
)
from .corrector_step import (
    GaussSeidelSolver,
    GSSolverSemiImplicitCorr,
    JacobiSolver,
)
from .corrector_v2 import BaseProjectionSolver
from .predictor_step import (
    EulerIntegrator,
    PredictorCorrectorIntegrator,
    RK4Integrator,
    SemiImplicitIntegrator,
)
from .rhs_discretization import (
    FiniteDifferenceDiscretizer,
    FiniteDifferenceUpwindDiscretizer,
    FiniteVolumeDiscretizer,
)

_NOT_IMPLEMENTED = "Direct instantiation of '{name}' is currently not supported. "


class GaussSeidelSolver:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="GaussSeidelSolver"))


class GSSolverSemiImplicitCorr:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            _NOT_IMPLEMENTED.format(name="GSSolverSemiImplicitCorr")
        )


class JacobiSolver:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED.format(name="JacobiSolver"))


__all__ = [
    "BaseProjectionSolver",
    "EulerIntegrator",
    "FiniteDifferenceDiscretizer",
    "FiniteDifferenceUpwindDiscretizer",
    "FiniteVolumeDiscretizer",
    "GSSolverSemiImplicitCorr",
    "GaussSeidelSolver",
    "JacobiSolver",
    "NavierStokesSolver2D",
    "PredictorCorrectorIntegrator",
    "RK4Integrator",
    "SemiImplicitIntegrator",
    "SpatialDiscretizationStrategy",
    "TimeIntegratorStrategy",
]
__version__ = "0.1.0"
