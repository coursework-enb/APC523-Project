
N_STEPS = 10000

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
benchmark_options = ["Taylor-Green Vortex", "Lid-Driven Cavity"]
nu_options = [1e-3, 1e-5]

# Run for all and get the combinations
