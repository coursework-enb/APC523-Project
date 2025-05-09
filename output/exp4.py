import time
import json
from numba import set_num_threads
set_num_threads(28)

from ns2d import (
    BaseProjectionSolver,
    FiniteDifferenceUpwindDiscretizer,
    SemiImplicitIntegrator,
    FiniteVolumeDiscretizer,
    EulerIntegrator,
    PredictorCorrectorIntegrator
)

BENCHMARK = "Taylor-Green Vortex"
END_TIME = 0.3         # Fixed physical time
FIXED_DT = True
CFL_BASED = False

NX = NY = 41
DX = DY = 2.0 / (NX - 1)
NU = 1e-3

dt_values = [0.001, 0.002, 0.004, 0.008, 0.012, 0.05]
results = []

for dt in dt_values:
    solver = BaseProjectionSolver(
        nx=NX,
        ny=NY,
        dx=DX,
        dy=DY,
        dt=dt,
        nu=NU,
        integrator= PredictorCorrectorIntegrator(),
        discrete_navier_stokes=FiniteVolumeDiscretizer(),
        fixed_dt=FIXED_DT,
    )

    try:
        print(f"Running dt = {dt:.4f}")
        start_time = time.perf_counter()
        _, _, error = solver.integrate(
            end_time=END_TIME,
            num_steps=None,
            benchmark=BENCHMARK,
            cfl_based=CFL_BASED,
        )
        compute_time = time.perf_counter() - start_time

        results.append({
            "solver": repr(solver),
            "dt": dt,
            "nu": NU,
            "end_time": END_TIME,
            "compute_time": compute_time,
            "error": error
        })

        print(f"✓ Completed in {compute_time:.2f}s | Error = {error:.2e}")

    except Exception as e:
        print(f"WARNING: Failed at dt = {dt}")
        results.append({
            "solver": repr(solver),
            "dt": dt,
            "nu": NU,
            "end_time": END_TIME,
            "compute_time": None,
            "error": str(e)
        })

with open("results_exp6_dt_sweep.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to 'results_exp6_dt_sweep.json'")
