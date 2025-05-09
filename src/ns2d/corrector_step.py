from typing import cast

import numpy as np
import pyamg
from numba import njit, prange
from scipy.sparse import diags, eye, kron

from .core import NavierStokesSolver2D
from .utils import Grid2D

# ------------------------
# Numerical Kernels
# ------------------------


@njit(parallel=True)
def divergence(u: Grid2D, v: Grid2D, dx: float, dy: float) -> Grid2D:
    """
    Compute the divergence of a vector field (u, v).
    Returns a 2D array of divergence values.
    """
    nx, ny = u.shape
    div = np.zeros_like(u)

    # Use second-order central differences for the interior points
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            div[i, j] = (u[i + 1, j] - u[i - 1, j]) / (2 * dx) + (
                v[i, j + 1] - v[i, j - 1]
            ) / (2 * dy)

    # Boundaries (one-sided differences)
    # Left boundary (i=0)
    for j in prange(1, ny - 1):
        div[0, j] = (u[1, j] - u[0, j]) / dx + (v[0, j + 1] - v[0, j - 1]) / (2 * dy)
    # Right boundary (i=nx-1)
    for j in prange(1, ny - 1):
        div[-1, j] = (u[-1, j] - u[-2, j]) / dx + (v[-1, j + 1] - v[-1, j - 1]) / (2 * dy)
    # Bottom boundary (j=0)
    for i in prange(1, nx - 1):
        div[i, 0] = (u[i + 1, 0] - u[i - 1, 0]) / (2 * dx) + (v[i, 1] - v[i, 0]) / dy
    # Top boundary (j=ny-1)
    for i in prange(1, nx - 1):
        div[i, -1] = (u[i + 1, -1] - u[i - 1, -1]) / (2 * dx) + (v[i, -1] - v[i, -2]) / dy
    # Corners (simplified)
    div[0, 0] = (u[1, 0] - u[0, 0]) / dx + (v[0, 1] - v[0, 0]) / dy
    div[0, -1] = (u[1, -1] - u[0, -1]) / dx + (v[0, -1] - v[0, -2]) / dy
    div[-1, 0] = (u[-1, 0] - u[-2, 0]) / dx + (v[-1, 1] - v[-1, 0]) / dy
    div[-1, -1] = (u[-1, -1] - u[-2, -1]) / dx + (v[-1, -1] - v[-1, -2]) / dy

    return div


@njit(parallel=True)
def velocity_correction(
    u_star: Grid2D, v_star: Grid2D, p: Grid2D, dx: float, dy: float, dt: float
) -> tuple[Grid2D, Grid2D]:
    """
    Correct the intermediate velocity using pressure gradient.
    Returns the corrected velocity fields (u, v).
    """
    nx, ny = u_star.shape
    u_corr = np.zeros_like(u_star)
    v_corr = np.zeros_like(v_star)

    # Interior points
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx)
            dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy)
            u_corr[i, j] = u_star[i, j] - dt * dp_dx
            v_corr[i, j] = v_star[i, j] - dt * dp_dy
    return u_corr, v_corr


@njit(parallel=True)
def apply_semi_implicit_diffusion(
    velocity: Grid2D, nu: float, dt: float, dx: float
) -> Grid2D:
    nx, ny = velocity.shape
    result = velocity.copy()
    alpha = nu * dt
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            lap = (
                velocity[i + 1, j]
                + velocity[i - 1, j]
                + velocity[i, j + 1]
                + velocity[i, j - 1]
                - 4 * velocity[i, j]
            ) / dx**2
            result[i, j] += alpha * lap
    return result


def semi_implicit_velocity_update(
    u_star: Grid2D, v_star: Grid2D, p: Grid2D, dx: float, dy: float, dt: float, nu: float
) -> tuple[Grid2D, Grid2D]:
    u_corr, v_corr = velocity_correction(u_star, v_star, p, dx, dy, dt)
    u_new = apply_semi_implicit_diffusion(u_corr, nu, dt, dx)
    v_new = apply_semi_implicit_diffusion(v_corr, nu, dt, dx)
    return u_new, v_new


# ------------------------
# Multigrid Poisson Solver
# ------------------------


@njit(fastmath=True)
def pressure_poisson_jacobi(
    p_initial: np.ndarray, b: np.ndarray, dx: float, dy: float, nit: int
) -> Grid2D:
    """
    Solves the 2D pressure Poisson equation using the Jacobi iterative method.
    ∇²p = b
    :return: The calculated pressure field after nit iterations.
    """
    ny, nx = p_initial.shape
    p = p_initial.copy()
    pn = np.empty_like(p)

    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    assert denom != 0

    term_b_multiplier = (dx2 * dy2) / denom

    for _ in range(nit):
        pn = p.copy()

        # Update interior points using values from pn
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                p[i, j] = (
                    (pn[i, j + 1] + pn[i, j - 1]) * dy2
                    + (pn[i + 1, j] + pn[i - 1, j]) * dx2
                ) / denom - term_b_multiplier * b[i, j]

    return p


def pressure_poisson_multigrid(
    rhs: Grid2D, dx: float, dy: float, smoother: str = "jacobi"
) -> Grid2D:
    """
    Solve ∇²p = rhs using multigrid method with chosen smoother.
    Returns the pressure field.
    """
    nx, ny = rhs.shape
    Tx = diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Ty = diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    A = kron(eye(ny), Tx) + kron(Ty, eye(nx))
    A = A.tocsr()
    b = rhs.ravel()

    if smoother == "jacobi":
        ml = pyamg.ruge_stuben_solver(
            A,
            presmoother=("jacobi", {"omega": 4.0 / 3.0}),
            postsmoother=("jacobi", {"omega": 4.0 / 3.0}),
        )
    elif smoother == "gauss_seidel":
        ml = pyamg.ruge_stuben_solver(
            A, presmoother="gauss_seidel", postsmoother="gauss_seidel"
        )
    else:
        raise ValueError("Unsupported smoother")

    x = ml.solve(b, tol=1e-8)
    p = x.reshape((nx, ny))

    return cast(Grid2D, p)


# ------------------------
# Solver Implementations
# ------------------------


class JacobiSolver(NavierStokesSolver2D):
    def solve_poisson(self) -> None:
        """
        Apply multigrid Poisson solver with Jacobi smoother to update pressure.
        """
        rhs = divergence(self.u, self.v, self.dx, self.dy) / self.dt

        if self.bc_case == 1:  # Periodic BC
            rhs[0, :] = rhs[-2, :]
            rhs[-1, :] = rhs[1, :]
            rhs[:, 0] = rhs[:, -2]
            rhs[:, -1] = rhs[:, 1]
        # Note: For case 2, we keep one-sided difference approximation

        self.p = pressure_poisson_multigrid(rhs, self.dx, self.dy, smoother="jacobi")
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        """
        Update velocity field using pressure gradient correction.
        """
        self.u, self.v = velocity_correction(
            self.u, self.v, self.p, self.dx, self.dy, self.dt
        )
        self._apply_bc(p=False)


class GaussSeidelSolver(NavierStokesSolver2D):
    def solve_poisson(self) -> None:
        """
        Apply multigrid Poisson solver with Gauss-Seidel smoother to update pressure.
        """
        rhs = divergence(self.u, self.v, self.dx, self.dy) / self.dt

        if self.bc_case == 1:  # Periodic BC
            rhs[0, :] = rhs[-2, :]
            rhs[-1, :] = rhs[1, :]
            rhs[:, 0] = rhs[:, -2]
            rhs[:, -1] = rhs[:, 1]
        # Note: For case 2, we keep one-sided difference approximation

        self.p = pressure_poisson_multigrid(
            rhs, self.dx, self.dy, smoother="gauss_seidel"
        )
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        """
        Update velocity field using pressure gradient correction.
        """
        self.u, self.v = velocity_correction(
            self.u, self.v, self.p, self.dx, self.dy, self.dt
        )
        self._apply_bc(p=False)


class GSSolverSemiImplicitCorr(NavierStokesSolver2D):
    def solve_poisson(self) -> None:
        """
        Apply multigrid Poisson solver with Gauss-Seidel smoother to update pressure,
        using semi-implicit correction instead.
        """
        rhs = divergence(self.u, self.v, self.dx, self.dy) / self.dt

        if self.bc_case == 1:  # Periodic BC
            rhs[0, :] = rhs[-2, :]
            rhs[-1, :] = rhs[1, :]
            rhs[:, 0] = rhs[:, -2]
            rhs[:, -1] = rhs[:, 1]
        # Note: For case 2, we keep one-sided difference approximation

        self.p = pressure_poisson_multigrid(
            rhs, self.dx, self.dy, smoother="gauss_seidel"
        )
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        """
        Update velocity field using pressure gradient correction.
        """
        self.u, self.v = semi_implicit_velocity_update(
            self.u, self.v, self.p, self.dx, self.dy, self.dt, self.nu
        )
        self._apply_bc(p=False)
