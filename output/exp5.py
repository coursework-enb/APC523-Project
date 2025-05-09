import time
import json
from numba import set_num_threads
set_num_threads(28)

from ns2d import (
    BaseProjectionSolver,
    FiniteDifferenceUpwindDiscretizer,
    SemiImplicitIntegrator,
)

BENCHMARK = "Taylor-Green Vortex"
NU = 1e-3
DT = 0.005
FIXED_DT = True
CFL_BASED = False
N_STEPS = 10000

results = []
for i in range(3, 10):
    N = 2**i
    dx = dy = 2.0 / (N - 1)

    solver = BaseProjectionSolver(
        nx=N,
        ny=N,
        dx=dx,
        dy=dy,
        dt=DT,
        nu=NU,
        integrator=SemiImplicitIntegrator(),
        discrete_navier_stokes=FiniteDifferenceUpwindDiscretizer(),
        fixed_dt=FIXED_DT,
    )

    try:
        print(f"Running N = {N}x{N} | dx = {dx:.5f}")
        start_time = time.perf_counter()
        _, _, error = solver.integrate(num_steps=N_STEPS, end_time=None, benchmark=BENCHMARK, cfl_based=CFL_BASED)
        compute_time = time.perf_counter() - start_time

        results.append({
            "solver": repr(solver),
            "N": N,
            "dx": dx,
            "dt": DT,
            "nu": NU,
            "fixed_dt": FIXED_DT,
            "num_steps": N_STEPS,
            "integrator": "SemiImplicitIntegrator",
            "discretizer": "FiniteDifferenceUpwindDiscretizer",
            "benchmark": BENCHMARK,
            "compute_time": compute_time,
            "error": error
        })

        print(f"✓ Completed in {compute_time:.2f}s | Error = {error:.2e}")

    except Exception as e:
        print(f"WARNING: Failed for N = {N} → {str(e)}")
        results.append({
            "solver": repr(solver),
            "N": N,
            "dx": dx,
            "dt": DT,
            "nu": NU,
            "fixed_dt": FIXED_DT,
            "num_steps": N_STEPS,
            "integrator": "SemiImplicitIntegrator",
            "discretizer": "FiniteDifferenceUpwindDiscretizer",
            "benchmark": BENCHMARK,
            "compute_time": None,
            "error": str(e)
        })

with open('results_exp5_performance.json', 'w') as f:
    json.dump(results, f, indent=4)
