import time
import itertools
import json

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

results = []
all_combinations = list(itertools.product(rhs_options, lhs_options, nu_options))
print(f"Number of configurations to run: {len(all_combinations)}")

for current_ns_discretizer, current_integrator, current_nu in all_combinations:
    solver = BaseProjectionSolver(
        nx=NX, ny=NY, dx=DX, dy=DY, dt=0.004, nu=current_nu,
        integrator=current_integrator,
        discrete_navier_stokes=current_ns_discretizer,
        fixed_dt=FIXED_DT
    )

    try:
        start_time = time.perf_counter()
        _, _, error = solver.integrate(
            num_steps=N_STEPS,
            end_time=None,
            benchmark=BENCHMARK,
            cfl_based=CFL_BASED,
        )
        compute_time = time.perf_counter() - start_time

        results.append({
            "solver": repr(solver),
            'discretizer': current_ns_discretizer.__class__.__name__,
            'integrator': current_integrator.__class__.__name__,
            'nu': current_nu,
            'compute_time': compute_time,
            'error': error
        })

    except Exception as e:
        print("WARNING: Current combination failed")
        results.append({
            "solver": repr(solver),
            'discretizer': current_ns_discretizer.__class__.__name__,
            'integrator': current_integrator.__class__.__name__,
            'nu': current_nu,
            'error': str(e),
            'compute_time': None
        })

with open('results_exp1.json', 'w') as f:
    json.dump(results, f, indent=4)
