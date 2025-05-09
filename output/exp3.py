from ns2d import (
    BaseProjectionSolver,
    FiniteDifferenceDiscretizer,
    FiniteDifferenceUpwindDiscretizer,
    FiniteVolumeDiscretizer,
    EulerIntegrator,
    PredictorCorrectorIntegrator,
    RK4Integrator,
    SemiImplicitIntegrator
)

BENCHMARK = "Taylor-Green Vortex"
N_STEPS = 10000
FIXED_DT = True
CFL_BASED = True
NX, NY = 41, 41
DX, DY = 2.0 / (NX - 1), 2.0 / (NY - 1)


rhs_options = [
    FiniteDifferenceDiscretizer(),
    FiniteDifferenceUpwindDiscretizer(),
    FiniteVolumeDiscretizer(),
]
lhs_options = [
    EulerIntegrator(),
    PredictorCorrectorIntegrator(),
    RK4Integrator(),
    SemiImplicitIntegrator(),
]
nu_options = [1e-3, 1e-5]

# Run for all combinations and get the compute time and error
