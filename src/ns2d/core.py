# TODO:
# - Make sure that boundary conditions are enforced consistently and coherently
# - Add external forces
# - Define a proper default initialization (when no benchmark) + manual init
# - For central differences make the choice of order more flexible like in ns2d/vorticity.py

# Diagnosis using
# - JacobiSolver with SemiImplicitIntegrator and FiniteDifferenceDiscretizer
# - TGV over 10000 steps

# We get an error for fixed dt, cfl-based dt, adaptive dt with either options
#   - fixed dt = 0.01, 0.001, 1e-05 or 1e-08 leads to pre-Poisson failure
#   - fixed cfl-based dt with target = 0.2 or 0.01 leads to failure
#   - adaptive dt with cfl_adapt (both default and 0.01 target) leads to pre-Poisson failure
#   - adaptive dt without cfl_adapt leads to pre-Poisson failure

# If now self.dt is NOT updated (in comment), hence when it does not apply to corrector step we get:
#   - adaptive dt without cfl-adaptive strategy works very well
#   - adaptive dt with cfl-adaptation and default target of 0.2 leads to pre-Poisson failure
#   - adaptive dt with cfl-adaptation and target of 0.01 runs without NaNs and provides a good solution
#   - fixed dt with cfl-based (target 0.01) dt runs without NaNs but does not give as good of a solution
#     and it fails for self.dt to small (1e-5) but works for higher self.dt !

# Note: all failures are within the first 100 steps and we seem to get the same with
# GaussSeidelSolver, SemiImplicitIntegrator, FiniteDifferenceUpwindDiscretizer


from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy import float64
from tqdm import tqdm

from .adaptive_time import adapt_time_step, cfl_adapt_time_step
from .benchmarks import (
    initialize_for_benchmark,
    validate_against_benchmark,
)
from .boundaries import apply_pressure_bc, apply_velocity_bc
from .safe_math import calculate_aggr_sq_speeds, calculate_max_velocity_magn
from .utils import Grid2D
from .vorticity import finite_difference_vorticity


class SpatialDiscretizationStrategy(ABC):
    @abstractmethod
    def __call__(
        self,
        u: Grid2D,
        v: Grid2D,
        p: Grid2D,
        dx: float,
        dy: float,
        nu: float,
        bc_case: int,
    ) -> tuple[Grid2D, Grid2D]:
        """Discretize spatial derivatives for momentum equations"""
        pass


class TimeIntegratorStrategy(ABC):
    @abstractmethod
    def advance_time(
        self,
        u: Grid2D,
        v: Grid2D,
        p_prev: Grid2D,
        dt: float,
        method: SpatialDiscretizationStrategy,  # inject dependency
        dx: float,
        dy: float,
        nu: float,  # with necessary inputs for call
        bc_case: int,
    ) -> tuple[Grid2D, Grid2D]:
        """Calculate intermediate velocities without pressure gradient"""
        pass


@dataclass
class NavierStokesSolver2D(ABC):
    """Context class that uses strategies for time integration, spatial discretization, and implements pressure solution"""

    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    nu: float
    integrator: TimeIntegratorStrategy
    discrete_navier_stokes: SpatialDiscretizationStrategy
    fixed_dt: bool = True
    min_dt: float = 1e-8
    max_dt: float = 1.0
    tol: float = 1e-1  # default: 10% rule
    target_CFL: float = 0.2
    strong_adaptive: bool = False
    failfast: bool = True
    u: Grid2D = field(init=False)
    v: Grid2D = field(init=False)
    p: Grid2D = field(init=False)
    bc_case: int = field(init=False)  # set when running initialize_fields:
    X: Grid2D = field(init=False)
    Y: Grid2D = field(init=False)

    def __post_init__(self) -> None:
        self.u = np.zeros((self.nx, self.ny), dtype=float64)
        self.v = np.zeros((self.nx, self.ny), dtype=float64)
        self.p = np.zeros((self.nx, self.ny), dtype=float64)

    def __repr__(self) -> str:
        is_abstract = ABC in self.__class__.__bases__
        poisson_solver = "abstract" if is_abstract else self.__class__.__name__
        return (
            f"NavierStokesSolver2D(adaptive={not self.fixed_dt}, "
            f"grid=({self.nx}x{self.ny}), "
            f"scheme={self.discrete_navier_stokes.__class__.__name__}, "
            f"integrator={self.integrator.__class__.__name__}, "
            f"poisson={poisson_solver})"
        )

    def _apply_bc(self, uv: bool = True, p: bool = True) -> None:
        if uv:
            self.u, self.v = apply_velocity_bc(self.u, self.v, self.bc_case)
        if p:
            self.p = apply_pressure_bc(self.p, self.bc_case)

    def _estimate_cfl(self) -> float:
        """Compute the maximum Courant-Friedrichs-Lewy (CFL) number"""
        cfl: float = np.maximum(
            self.u * self.dt / self.dx, self.v * self.dt / self.dy
        ).max()
        return cfl

    def set_time_integrator(self, integrator: TimeIntegratorStrategy) -> None:
        """Change the time integration strategy at runtime"""
        self.integrator = integrator

    def set_spatial_discretizer(self, discretizer: SpatialDiscretizationStrategy) -> None:
        """Change the spatial discretization strategy at runtime"""
        self.discrete_navier_stokes = discretizer

    def initialize_fields(self, benchmark: str | None) -> None:
        """Initialize velocity and pressure fields."""
        if benchmark is not None:
            self.X, self.Y, self.u, self.v, self.p, self.bc_case = (
                initialize_for_benchmark(benchmark, self.nx, self.ny)
            )
        else:
            self.bc_case = 0
            print("WARNING: No benchmark")
            self.u.fill(0.0)
            self.v.fill(0.0)
            self.p.fill(0.0)

    @abstractmethod
    def solve_poisson(self) -> None:
        """Solve the pressure Poisson equation"""
        raise NotImplementedError("Poisson solver not implemented")

    @abstractmethod
    def update_velocity(self) -> None:
        """Update velocity fields based on the pressure correction from the Poisson solver."""
        raise NotImplementedError("Velocity update function not implemented")

    def compute_kinetic_energy(self) -> float:
        """
        Compute the kinetic energy of the velocity field, scaled by grid cell area.

        :return: Kinetic energy value
        """
        cell_area = self.dx * self.dy
        sum_sq = calculate_aggr_sq_speeds(self.u, self.v)
        ke: float = 0.5 * cell_area * sum_sq
        return ke

    def compute_vorticity(self, order: int = 2) -> Grid2D:
        """Compute vorticity for validation purposes.

        :return: A 2D list representing the vorticity field.
        """
        result: Grid2D = finite_difference_vorticity(
            self.u, self.v, self.dx, self.dy, order=order
        )
        # We could also compute it using the stream function field (from below) and computing the Laplacian of psi using finite differences

        if self.bc_case == 1:  # Periodic BC
            result[0, :] = result[-2, :]
            result[-1, :] = result[1, :]
            result[:, 0] = result[:, -2]
            result[:, -1] = result[:, 1]

        # Note: For case 2, we approximate vorticity even at last boundary cells using one-sided differences,
        # but a more accurate approach would involve computing wall vorticity explicitly to ensure
        # consistency with the no-slip condition.

        return result

    def solve_stream_function(self) -> Grid2D:
        """
        Compute the stream function by solving the Poisson equation -∇²ψ = ω using the class's Poisson solver.
        Vorticity is used as the source term.

        :return: A 2D array representing the stream function field
        """
        vorticity = self.compute_vorticity()

        temp_p = self.p.copy()
        self.p = vorticity
        self.solve_poisson()
        stream_func = self.p.copy()

        # Restore original pressure field
        # TODO: revise solve_poisson such that we can input/output the pressure without mutation (safer)
        self.p = temp_p

        if self.bc_case == 1:  # Periodic BC
            stream_func[0, :] = stream_func[-2, :]
            stream_func[-1, :] = stream_func[1, :]
            stream_func[:, 0] = stream_func[:, -2]
            stream_func[:, -1] = stream_func[:, 1]
        elif self.bc_case == 2:  # No normal flow through boundaries
            stream_func[:, 0] = 0  # Bottom wall
            stream_func[:, -1] = 0  # Top wall
            stream_func[0, :] = 0  # Left wall
            stream_func[-1, :] = 0  # Right wall

        return stream_func

    def validate(
        self, benchmark: str, current_time: float, verbose: bool = False
    ) -> float:
        """
        This method compares the solver's output (e.g., kinetic energy for Taylor-Green Vortex or minimum stream function for Lid-Driven Cavity) against reference or analytical solutions for the specified benchmark problem.

        :param benchmark: The benchmark problem to validate against
        """
        # Don't compute kinetic energy and stream when not needed
        if benchmark == "Taylor-Green Vortex":
            ke_simulated = self.compute_kinetic_energy()
            stream_func = np.zeros_like(self.u)
        elif benchmark == "Lid-Driven Cavity":
            ke_simulated = 0.0
            stream_func = self.solve_stream_function()
        # TODO: make it cleaner

        error: float = validate_against_benchmark(
            benchmark,
            self.dx,
            self.dy,
            self.X,
            self.Y,
            self.nu,
            current_time,
            ke_simulated,
            stream_func,
            verbose,
        )
        # Optional: Add visual for the error at each cell when Taylor-Green Vortex
        return error

    def _cfl_time(self) -> float:
        """Provides the new time step purely based on CFL"""
        max_velocity = calculate_max_velocity_magn(self.u, self.v)

        if np.isclose(max_velocity, 0.0):
            return self.max_dt

        h_min = min(self.dx, self.dy)

        dt: float = self.target_CFL * h_min / max_velocity
        dt = max(min(dt, self.max_dt), self.min_dt)

        return dt

    def _check_current_sol(self, hard: bool = False) -> tuple[bool, str]:
        """Checks solution's validity at runtime"""
        error = False
        messages = []

        if np.any(np.isnan(self.u)):
            messages.append("  - u contains NaN values")
            error = True
        if np.any(np.isinf(self.u)):
            messages.append("  - u contains Inf values")
            error = True
        if np.any(np.isnan(self.v)):
            messages.append("  - v contains NaN values")
            error = True
        if np.any(np.isinf(self.v)):
            messages.append("  - v contains Inf values")
            error = True
        if np.any(np.isnan(self.p)):
            messages.append("  - p contains NaN values")
            error = True
        if np.any(np.isinf(self.p)):
            messages.append("  - p contains Inf values")
            error = True

        if hard:
            ke = self.compute_kinetic_energy()
            stream = self.solve_stream_function()

            if np.isnan(ke):
                messages.append("  - kinetic energy is NaN")
                error = True
            if np.isinf(ke):
                messages.append("  - kinetic energy is Inf")
                error = True
            if np.any(np.isnan(stream)):
                messages.append("  - stream function contains NaN values")
                error = True
            if np.any(np.isinf(stream)):
                messages.append("  - stream function contains Inf values")
                error = True

        if error:
            error_message = "\n".join(messages)
            return False, error_message
        else:
            return True, "Solution is valid."

    def _fail_fast(
        self,
        step: int | None,
        prepend_message: str = "Validation failed with the following error(s):",
        append_message: str = "",
    ) -> None:
        if self.failfast and (step is None or step % 100 == 0):
            is_valid, message = self._check_current_sol()
            if not is_valid:
                raise RuntimeError(
                    prepend_message
                    + "\n"
                    + message
                    + "\n"
                    + f"with dt={self.dt} and CFL={self._estimate_cfl()}"
                    + append_message
                )
        return None

    def integrate(
        self,
        num_steps: int | None = None,
        end_time: float | None = 2.5,
        benchmark: str | None = "Lid-Driven Cavity",
        cfl_based: bool = True,
        cfl_adapt: bool = False,
    ) -> tuple[list[float], list[float], float] | tuple[list[float], list[float]]:
        """Run the simulation for a specified number of time steps.

        :params: Number of time steps or integration time, dt update strategy and benchmark
        :return: Serializable time values, CFL values and error
        """
        if num_steps is None and end_time is None:
            raise ValueError("Needs either num_steps or end_time")

        self.initialize_fields(benchmark)

        current_time = 0.0
        step = 0
        current_dt = self.dt

        max_steps: int | float = num_steps if num_steps is not None else float("inf")
        end_time = end_time if end_time is not None else float("inf")

        cfl_values = []
        time_values = []

        # If num_steps is provided, use it as the total; otherwise, estimate based on end_time and dt
        total_steps = num_steps if num_steps is not None else int(end_time / self.dt) + 1
        progress_bar = tqdm(total=total_steps, desc="Simulation Progress", unit="steps")

        while current_time < end_time and step < max_steps:
            # Ensure we don't overshoot the end time
            current_dt = min(current_dt, end_time - current_time)

            u_prev = self.u.copy()
            v_prev = self.v.copy()

            # Perform a tentative step
            self.u, self.v = self.integrator.advance_time(
                self.u,
                self.v,
                self.p,
                current_dt,
                self.discrete_navier_stokes,
                self.dx,
                self.dy,
                self.nu,
                self.bc_case,
            )

            self._fail_fast(
                step,
                prepend_message=f"Validation failed at step {step} pre-Poisson with the following error(s):",
                append_message=f" and current_dt={current_dt}",
            )

            # Enforce incompressibility
            self.solve_poisson()
            self.update_velocity()

            # If adaptive time stepping is enabled, check if the step is acceptable
            current_cfl = self._estimate_cfl()
            if not self.fixed_dt:
                if cfl_adapt:
                    # DEBUG: Produces NaN
                    dt_new, accept = cfl_adapt_time_step(
                        current_cfl, current_dt, self.min_dt, self.max_dt, self.target_CFL
                    )
                    # raise NotImplementedError(
                    #     "CFL-adaptive time step needs to be debugged"
                    # )
                else:
                    dt_new, accept = adapt_time_step(
                        u_prev,
                        v_prev,
                        self.u,
                        self.v,
                        current_dt,
                        self.min_dt,
                        self.max_dt,
                        self.tol,
                    )
                if self.strong_adaptive and not accept:
                    # If strong adaptive is enabled and step is rejected, revert and retry
                    self.u = u_prev
                    self.v = v_prev
                    current_dt = dt_new
                    continue
                current_dt = dt_new
            elif cfl_based:
                current_dt = self._cfl_time()
            # Else keep current_dt

            # Record
            cfl_values.append(current_cfl)
            time_values.append(current_time)

            self._fail_fast(
                step,
                prepend_message=f"Validation failed at step {step} post-Poisson with the following error(s):",
                append_message=f" and current_dt={current_dt}",
            )

            # Advance time and step counter
            current_time += current_dt
            step += 1

            progress_bar.update(1)

            self.dt = current_dt  # Issue is exactly here, uncommenting it leads to systematic failure!
            # print(f"Currend dt: {current_dt} / self.dt: {self.dt} / CFL {current_cfl}")

        progress_bar.close()

        if benchmark == "Lid-Driven Cavity":
            error_ldc: float = self.validate(benchmark, end_time, verbose=True)
            return time_values, cfl_values, error_ldc
        elif benchmark == "Taylor-Green Vortex":
            error_tgv: float = self.validate(benchmark, current_time)
            return time_values, cfl_values, error_tgv
        else:
            return time_values, cfl_values
