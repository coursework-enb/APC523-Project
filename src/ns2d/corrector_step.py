# TODO: Check boundary conditions for divergence

from typing import cast

import numpy as np
import pyamg
from scipy.sparse import diags, eye, kron

from .core import NavierStokesSolver2D
from .utils import Grid2D

# ------------------------
# Numerical Kernels
# ------------------------


def divergence(u: Grid2D, v: Grid2D, dx: float, dy: float) -> Grid2D:
    nx, ny = u.shape
    div = np.zeros_like(u)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            div[i, j] = (u[i + 1, j] - u[i - 1, j]) / (2 * dx) + (
                v[i, j + 1] - v[i, j - 1]
            ) / (2 * dy)
    return div


def velocity_correction(
    u_star: Grid2D, v_star: Grid2D, p: Grid2D, dx: float, dy: float, dt: float
) -> tuple[Grid2D, Grid2D]:
    nx, ny = u_star.shape
    u_corr = np.zeros_like(u_star)
    v_corr = np.zeros_like(v_star)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx)
            dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy)
            u_corr[i, j] = u_star[i, j] - dt * dp_dx
            v_corr[i, j] = v_star[i, j] - dt * dp_dy
    return u_corr, v_corr


# ------------------------
# Multigrid Poisson Solver
# ------------------------


def pressure_poisson_multigrid(
    rhs: Grid2D, dx: float, dy: float, smoother: str = "jacobi"
) -> Grid2D:
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
        rhs = divergence(self.u, self.v, self.dx, self.dy) / self.dt
        self.p = pressure_poisson_multigrid(rhs, self.dx, self.dy, smoother="jacobi")
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        self.u, self.v = velocity_correction(
            self.u, self.v, self.p, self.dx, self.dy, self.dt
        )
        self._apply_bc(p=False)


class GaussSeidelSolver(NavierStokesSolver2D):
    def solve_poisson(self) -> None:
        rhs = divergence(self.u, self.v, self.dx, self.dy) / self.dt
        self.p = pressure_poisson_multigrid(
            rhs, self.dx, self.dy, smoother="gauss_seidel"
        )
        self._apply_bc(uv=False)

    def update_velocity(self) -> None:
        self.u, self.v = velocity_correction(
            self.u, self.v, self.p, self.dx, self.dy, self.dt
        )
        self._apply_bc(p=False)
