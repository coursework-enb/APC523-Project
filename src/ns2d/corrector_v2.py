import numpy as np
from numba import njit, prange

from .boundaries import apply_pressure_bc
from .core import NavierStokesSolver2D
from .utils import Grid2D


@njit(parallel=True, fastmath=True)
def build_rhs(u_star: Grid2D, v_star: Grid2D, dt: float, dx: float, dy: float) -> Grid2D:
    """Calculates the Right-Hand Side (RHS) of the Pressure Poisson Equation (PPE)."""
    nx, ny = u_star.shape
    rhs = np.zeros_like(u_star)

    # Compute for interior points
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            # Second-order central differences for velocity derivatives
            du_dx = (u_star[i + 1, j] - u_star[i - 1, j]) / (2 * dx)
            dv_dy = (v_star[i, j + 1] - v_star[i, j - 1]) / (2 * dy)
            du_dy = (u_star[i, j + 1] - u_star[i, j - 1]) / (2 * dy)
            dv_dx = (v_star[i + 1, j] - v_star[i - 1, j]) / (2 * dx)

            # RHS formula for the PPE
            rhs[i, j] = (
                (1 / dt * (du_dx + dv_dy)) - (du_dx**2) - (2 * du_dy * dv_dx) - (dv_dy**2)
            )
    return rhs


@njit(parallel=True, fastmath=True)
def pressure_poisson_jacobi(
    p_initial: Grid2D,
    source_term: Grid2D,
    dx: float,
    dy: float,
    bc_case: int,
    num_iterations: int,
) -> Grid2D:
    """Solves the Pressure Poisson Equation (∇²p = source_term) using the Jacobi iterative method."""
    nx, ny = p_initial.shape
    p_current = p_initial.copy()
    pn = np.empty_like(p_current)

    dx2 = dx**2
    dy2 = dy**2
    denominator_poisson = 2 * (dx2 + dy2)
    multiplier_poisson = (dx2 * dy2) / denominator_poisson

    # Jacobi iterations
    for _ in range(num_iterations):
        pn = p_current.copy()

        # Update interior points using second-order central difference (five-point stencil)
        for i in prange(1, nx - 1):
            for j in prange(1, ny - 1):
                p_current[i, j] = (
                    (pn[i + 1, j] + pn[i - 1, j]) * dy2
                    + (pn[i, j + 1] + pn[i, j - 1]) * dx2
                ) / denominator_poisson - multiplier_poisson * source_term[i, j]

        # Apply pressure boundary conditions at each iteration
        p_current = apply_pressure_bc(p_current, bc_case)

    return p_current


@njit(parallel=True, fastmath=True)
def velocity_correction(
    u_intermediate: Grid2D,
    v_intermediate: Grid2D,
    p_field: Grid2D,
    dt: float,
    dx: float,
    dy: float,
) -> tuple[Grid2D, Grid2D]:
    """Corrects the intermediate velocity field using the calculated pressure gradient."""
    nx, ny = u_intermediate.shape
    u_corrected = u_intermediate.copy()
    v_corrected = v_intermediate.copy()

    # Calculate pressure gradient and update velocity for interior points
    for i in prange(1, nx - 1):
        for j in prange(1, ny - 1):
            # Second-order central differences for pressure gradients
            dp_dx = (p_field[i + 1, j] - p_field[i - 1, j]) / (2 * dx)
            dp_dy = (p_field[i, j + 1] - p_field[i, j - 1]) / (2 * dy)

            # Update velocities
            u_corrected[i, j] = u_intermediate[i, j] - dt * dp_dx
            v_corrected[i, j] = v_intermediate[i, j] - dt * dp_dy

    return u_corrected, v_corrected


class BaseProjectionSolver(NavierStokesSolver2D):
    def solve_poisson(self, num_iterations: int = 50) -> None:
        """
        Calculates the RHS of the Poisson equation and solves for pressure, updating self.p.
        Uses Numba-jitted functions for core computations.
        :param num_iterations: Number of Jacobi iterations.
        """
        source_term = build_rhs(self.u, self.v, self.dt, self.dx, self.dy)
        self.p = pressure_poisson_jacobi(
            self.p, source_term, self.dx, self.dy, self.bc_case, num_iterations
        )
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        """
        Corrects the intermediate velocity field using the calculated pressure gradient.
        Updates self.u and self.v in place. Uses Numba-jitted function.
        """
        self.u, self.v = velocity_correction(
            self.u, self.v, self.p, self.dt, self.dx, self.dy
        )
        self._apply_bc(p=False)
