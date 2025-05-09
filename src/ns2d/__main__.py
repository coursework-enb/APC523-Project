# mypy: ignore-errors

import json
import logging
import os
import time
from datetime import datetime
from itertools import product

import numpy as np

from ns2d import (
    EulerIntegrator,
    FiniteDifferenceDiscretizer,
    FiniteDifferenceUpwindDiscretizer,
    FiniteVolumeDiscretizer,
    GaussSeidelSolver,
    # JacobiSolver,  # TO DO: Add one such case, or do it individually
    PredictorCorrectorIntegrator,
    RK4Integrator,
    SemiImplicitIntegrator,
)

nx, ny = 41, 41
dx, dy = 2.0 / (nx - 1), 2.0 / (ny - 1)


def main():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f"simulation_run_{timestamp_str}.log"
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Starting new simulation script execution: {timestamp_str}")
    logging.info(f"Text logs will be saved to: {log_filename}")

    # --- Define possible parameter states ---
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

    fixed_options = [False, True]
    nu_options = [1e-3, 1e-5]
    dts_options = [0.005, 0.0075, 0.01]

    # --- Define configurations ---
    configs = [
        [
            rhs_options[0],
            lhs_options[2],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[2],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[0],
            lhs_options[1],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[1],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[0],
            lhs_options[0],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[0],
            nu_options[0],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[0],
            lhs_options[2],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[2],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[0],
            lhs_options[1],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[1],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[0],
            lhs_options[0],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
        [
            rhs_options[2],
            lhs_options[0],
            nu_options[1],
            fixed_options[0],
            dts_options[0],
            benchmark_options[1],
        ],
    ]
    for r_choice, l_choice, n_choice, f_choice, dt_choice, b_choice in product(
        [rhs_options[1]],
        [lhs_options[1], lhs_options[2]],
        nu_options,
        fixed_options,
        dts_options,
        benchmark_options,
    ):
        configs.append([r_choice, l_choice, n_choice, f_choice, dt_choice, b_choice])

    num_configs = len(configs)
    print(f"Number of configurations to run: {num_configs}")
    logging.info(f"Total number of configurations to run: {num_configs}")

    results_dir = f"simulation_results_{timestamp_str}"
    os.makedirs(results_dir, exist_ok=True)
    logging.info(
        f"Detailed simulation results (JSON files) will be saved in directory: {results_dir}"
    )

    # --- Simulation loop ---
    for i, config_params in enumerate(configs):
        discrete_navier_stokes = config_params[0]
        integrator = config_params[1]
        viscosity = config_params[2]
        fixed_dt_bool = config_params[3]
        current_dt_val = config_params[4]
        benchmark_val = config_params[5]

        logging.info(
            f"Running config {i + 1}/{num_configs}: dt={current_dt_val}, nu={viscosity}, "
            f"fixed_dt={fixed_dt_bool}, benchmark='{benchmark_val}', "
            f"integrator={integrator.__class__.__name__}, "
            f"discretizer={discrete_navier_stokes.__class__.__name__}"
        )

        solver = GaussSeidelSolver(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            dt=current_dt_val,
            nu=viscosity,
            integrator=integrator,
            discrete_navier_stokes=discrete_navier_stokes,
            fixed_dt=fixed_dt_bool,
        )

        if benchmark_val == "Taylor-Green Vortex":
            current_end_time = 0.5
        if benchmark_val == "Lid-Driven Cavity":
            current_end_time = 2.5

        start_time = time.perf_counter()
        try:
            time_values, cfl_values, error = solver.integrate(
                num_steps=None, end_time=current_end_time, benchmark=benchmark_val
            )
            vorticity = solver.compute_vorticity()
            compute_time = time.perf_counter() - start_time
            current_error = error

            if cfl_values is not None and cfl_values.size > 0:
                fraction_cfl_over_1 = np.sum(cfl_values > 1) / cfl_values.size
                mean_cfl = np.mean(cfl_values)
                cfl_stats = (fraction_cfl_over_1, mean_cfl)
            else:
                cfl_stats = (None, None)

            logging.info(
                f"Finished config {i + 1}: Compute time={compute_time:.2f}s, Final error={current_error:.3e}"
            )

            config_run_data = {
                "config_index": i + 1,
                "parameters": {
                    "dt": current_dt_val,
                    "nu": viscosity,
                    "fixed_dt": fixed_dt_bool,
                    "benchmark": benchmark_val,
                    "integrator": integrator.__class__.__name__,
                    "discretizer": discrete_navier_stokes.__class__.__name__,
                },
                "solver": repr(solver),
                "results": {
                    "time_values": time_values,
                    "cfl_stats": cfl_stats,
                    "final_error": current_error,
                    "compute_time_seconds": compute_time,
                    "vorticity": vorticity.tolist(),
                },
            }

            config_json_filename = os.path.join(
                results_dir, f"config_{i + 1:03d}_results.json"
            )
            with open(config_json_filename, "w") as f_json:
                json.dump(config_run_data, f_json, indent=4)
            logging.info(
                f"Saved detailed results for config {i + 1} to: {config_json_filename}"
            )

        except Exception as e:
            logging.error(f"Error running or saving config {i + 1}: {e}", exc_info=True)

    logging.info(f"All {num_configs} configurations processed.")
    logging.info(f"Simulation script execution {timestamp_str} completed.")
    print(f"\nSimulation run {timestamp_str} completed.")
    print(f"Text logs saved to: {log_filename}")
    print(f"Detailed results (JSON files) saved in directory: {results_dir}")


if __name__ == "__main__":
    main()
