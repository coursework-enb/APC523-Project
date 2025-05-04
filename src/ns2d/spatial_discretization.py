from typing import cast

import numpy as np
from numba import njit, prange

from ns2d import SpatialDiscretizationStrategy

from .utils import Grid2D


@njit(parallel=True)
def compute_fd_rhs(
    u: Grid2D,
    v: Grid2D,
    p: Grid2D,
    dx: float,
    dy: float,
    nu: float,
    include_pressure: bool = True,
) -> tuple[Grid2D, Grid2D]:
    """
    Compute the right-hand side of the momentum equations using central finite difference.
    Numba-optimized implementation.
    """
    nx, ny = u.shape
    du_dt = np.zeros_like(u)
    dv_dt = np.zeros_like(v)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            du_dx = (u[i + 1, j] - u[i - 1, j]) / (2 * dx)
            du_dy = (u[i, j + 1] - u[i, j - 1]) / (2 * dy)
            dv_dx = (v[i + 1, j] - v[i - 1, j]) / (2 * dx)
            dv_dy = (v[i, j + 1] - v[i, j - 1]) / (2 * dy)

            conv_u = u[i, j] * du_dx + v[i, j] * du_dy
            conv_v = u[i, j] * dv_dx + v[i, j] * dv_dy

            lap_u = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 + (
                u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]
            ) / dy**2
            lap_v = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2 + (
                v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
            ) / dy**2

            dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx) if include_pressure else 0.0
            dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy) if include_pressure else 0.0

            du_dt[i, j] = -conv_u + nu * lap_u - dp_dx
            dv_dt[i, j] = -conv_v + nu * lap_v - dp_dy

    return du_dt, dv_dt


class FiniteDifferenceDiscretizer(SpatialDiscretizationStrategy):
    def __call__(
        self, u: Grid2D, v: Grid2D, p: Grid2D, dx: float, dy: float, nu: float
    ) -> tuple[Grid2D, Grid2D]:
        """
        Discretize spatial derivatives for momentum equations using central finite difference.

        Args:
            u (Grid2D): Velocity component in x-direction.
            v (Grid2D): Velocity component in y-direction.
            p (Grid2D): Pressure field.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Time derivatives (du_dt, dv_dt) for u and v components.
        """
        result = compute_fd_rhs(u, v, p, dx, dy, nu, include_pressure=False)
        return cast(tuple[Grid2D, Grid2D], result)


@njit(parallel=True)
def compute_fd_upwind_rhs(
    u: Grid2D,
    v: Grid2D,
    p: Grid2D,
    dx: float,
    dy: float,
    nu: float,
    include_pressure: bool = True,
) -> tuple[Grid2D, Grid2D]:
    """
    Compute the right-hand side of the momentum equations using upwind finite difference for convection.
    Numba-optimized implementation.
    """
    nx, ny = u.shape
    du_dt = np.zeros_like(u)
    dv_dt = np.zeros_like(v)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            if u[i, j] > 0:
                du_dx = (u[i, j] - u[i - 1, j]) / dx
                dv_dx = (v[i, j] - v[i - 1, j]) / dx
            else:
                du_dx = (u[i + 1, j] - u[i, j]) / dx
                dv_dx = (v[i + 1, j] - v[i, j]) / dx
            if v[i, j] > 0:
                du_dy = (u[i, j] - u[i, j - 1]) / dy
                dv_dy = (v[i, j] - v[i, j - 1]) / dy
            else:
                du_dy = (u[i, j + 1] - u[i, j]) / dy
                dv_dy = (v[i, j + 1] - v[i, j]) / dy

            conv_u = u[i, j] * du_dx + v[i, j] * du_dy
            conv_v = u[i, j] * dv_dx + v[i, j] * dv_dy

            lap_u = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 + (
                u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]
            ) / dy**2
            lap_v = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2 + (
                v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
            ) / dy**2

            dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx) if include_pressure else 0.0
            dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy) if include_pressure else 0.0

            du_dt[i, j] = -conv_u + nu * lap_u - dp_dx
            dv_dt[i, j] = -conv_v + nu * lap_v - dp_dy

    return du_dt, dv_dt


class FiniteDifferenceUpwindDiscretizer(SpatialDiscretizationStrategy):
    def __call__(
        self, u: Grid2D, v: Grid2D, p: Grid2D, dx: float, dy: float, nu: float
    ) -> tuple[Grid2D, Grid2D]:
        """
        Discretize spatial derivatives for momentum equations using upwind finite difference for convection.

        Args:
            u (Grid2D): Velocity component in x-direction.
            v (Grid2D): Velocity component in y-direction.
            p (Grid2D): Pressure field.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Time derivatives (du_dt, dv_dt) for u and v components.
        """
        result = compute_fd_upwind_rhs(u, v, p, dx, dy, nu, include_pressure=False)
        return cast(tuple[Grid2D, Grid2D], result)


@njit(parallel=True)
def compute_fv_rhs(
    u: Grid2D,
    v: Grid2D,
    p: Grid2D,
    dx: float,
    dy: float,
    nu: float,
    include_pressure: bool = True,
) -> tuple[Grid2D, Grid2D]:
    """
    Compute the right-hand side of the momentum equations using finite volume method.
    Numba-optimized implementation.
    """
    nx, ny = u.shape
    du_dt = np.zeros_like(u)
    dv_dt = np.zeros_like(v)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            u_e = 0.5 * (u[i + 1, j] + u[i, j])
            u_w = 0.5 * (u[i, j] + u[i - 1, j])
            v_n = 0.5 * (v[i, j + 1] + v[i, j])
            v_s = 0.5 * (v[i, j] + v[i, j - 1])
            v_e = 0.5 * (v[i + 1, j] + v[i, j])
            v_w = 0.5 * (v[i - 1, j] + v[i, j])

            flux_conv_u = (u_e**2 - u_w**2) / dx + (
                u[i, j + 1] * v_n - u[i, j - 1] * v_s
            ) / dy
            flux_conv_v = (u_e * v_e - u_w * v_w) / dx + (v_n**2 - v_s**2) / dy

            lap_u = (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx**2 + (
                u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]
            ) / dy**2
            lap_v = (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx**2 + (
                v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]
            ) / dy**2

            dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2 * dx) if include_pressure else 0.0
            dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2 * dy) if include_pressure else 0.0

            du_dt[i, j] = -flux_conv_u + nu * lap_u - dp_dx
            dv_dt[i, j] = -flux_conv_v + nu * lap_v - dp_dy

    return du_dt, dv_dt


class FiniteVolumeDiscretizer(SpatialDiscretizationStrategy):
    def __call__(
        self, u: Grid2D, v: Grid2D, p: Grid2D, dx: float, dy: float, nu: float
    ) -> tuple[Grid2D, Grid2D]:
        """
        Discretize spatial derivatives for momentum equations using finite volume method.

        Args:
            u (Grid2D): Velocity component in x-direction.
            v (Grid2D): Velocity component in y-direction.
            p (Grid2D): Pressure field.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Time derivatives (du_dt, dv_dt) for u and v components.
        """
        result = compute_fv_rhs(u, v, p, dx, dy, nu, include_pressure=False)
        return cast(tuple[Grid2D, Grid2D], result)
