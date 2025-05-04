from typing import cast

import numpy as np
from numba import njit, prange

from ns2d import SpatialDiscretizationStrategy, TimeIntegratorStrategy

from .utils import Grid2D

@njit(parallel=True)
def euler_step(
    u: Grid2D, v: Grid2D, du_dt: Grid2D, dv_dt: Grid2D, dt: float
) -> tuple[Grid2D, Grid2D]:
    """
    Perform a forward Euler time step for velocity fields.

    Args:
        u (Grid2D): Current x-velocity field.
        v (Grid2D): Current y-velocity field.
        du_dt (Grid2D): Time derivative of x-velocity.
        dv_dt (Grid2D): Time derivative of y-velocity.
        dt (float): Time step size.

    Returns:
        tuple[Grid2D, Grid2D]: Updated (u, v) velocity fields.
    """
    nx, ny = u.shape
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = u[i, j] + dt * du_dt[i, j]
            v_new[i, j] = v[i, j] + dt * dv_dt[i, j]

    return u_new, v_new


@njit(parallel=True)
def rk4_step(
    u: Grid2D,
    v: Grid2D,
    k1_u: Grid2D,
    k1_v: Grid2D,
    k2_u: Grid2D,
    k2_v: Grid2D,
    k3_u: Grid2D,
    k3_v: Grid2D,
    k4_u: Grid2D,
    k4_v: Grid2D,
    dt: float,
) -> tuple[Grid2D, Grid2D]:
    """
    Perform a fourth-order Runge-Kutta (RK4) time step for velocity fields.

    Args:
        u (Grid2D): Current x-velocity field.
        v (Grid2D): Current y-velocity field.
        k1_u, k2_u, k3_u, k4_u (Grid2D): RK stages for x-velocity derivative.
        k1_v, k2_v, k3_v, k4_v (Grid2D): RK stages for y-velocity derivative.
        dt (float): Time step size.

    Returns:
        tuple[Grid2D, Grid2D]: Updated (u, v) velocity fields.
    """
    nx, ny = u.shape
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = u[i, j] + (dt / 6.0) * (
                k1_u[i, j] + 2 * k2_u[i, j] + 2 * k3_u[i, j] + k4_u[i, j]
            )
            v_new[i, j] = v[i, j] + (dt / 6.0) * (
                k1_v[i, j] + 2 * k2_v[i, j] + 2 * k3_v[i, j] + k4_v[i, j]
            )

    return u_new, v_new


class EulerIntegrator(TimeIntegratorStrategy):
    def advance_time(
        self,
        u: Grid2D,
        v: Grid2D,
        p_prev: Grid2D,
        dt: float,
        method: SpatialDiscretizationStrategy,
        dx: float,
        dy: float,
        nu: float,
    ) -> tuple[Grid2D, Grid2D]:
        """
        Advance velocity fields in time using forward Euler method without pressure gradient.

        Args:
            u (Grid2D): Current x-velocity field.
            v (Grid2D): Current y-velocity field.
            p_prev (Grid2D): Previous pressure field (unused in predictor step).
            dt (float): Time step size.
            method (SpatialDiscretizationStrategy): Spatial discretization method.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Intermediate velocity fields (u, v).
        """
        # Compute RHS without pressure gradient
        du_dt, dv_dt = method(u, v, p_prev, dx, dy, nu)
        result = euler_step(u, v, du_dt, dv_dt, dt)
        return cast(tuple[Grid2D, Grid2D], result)


class RK4Integrator(TimeIntegratorStrategy):
    def advance_time(
        self,
        u: Grid2D,
        v: Grid2D,
        p_prev: Grid2D,
        dt: float,
        method: SpatialDiscretizationStrategy,
        dx: float,
        dy: float,
        nu: float,
    ) -> tuple[Grid2D, Grid2D]:
        """
        Advance velocity fields in time using fourth-order Runge-Kutta (RK4) method without pressure gradient.

        Args:
            u (Grid2D): Current x-velocity field.
            v (Grid2D): Current y-velocity field.
            p_prev (Grid2D): Previous pressure field (unused in predictor step).
            dt (float): Time step size.
            method (SpatialDiscretizationStrategy): Spatial discretization method.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Intermediate velocity fields (u, v).
        """
        # Stage 1
        k1_u, k1_v = method(u, v, p_prev, dx, dy, nu)
        u1 = u + 0.5 * dt * k1_u
        v1 = v + 0.5 * dt * k1_v

        # Stage 2
        k2_u, k2_v = method(u1, v1, p_prev, dx, dy, nu)
        u2 = u + 0.5 * dt * k2_u
        v2 = v + 0.5 * dt * k2_v

        # Stage 3
        k3_u, k3_v = method(u2, v2, p_prev, dx, dy, nu)
        u3 = u + dt * k3_u
        v3 = v + dt * k3_v

        # Stage 4
        k4_u, k4_v = method(u3, v3, p_prev, dx, dy, nu)

        # Final update
        result = rk4_step(u, v, k1_u, k1_v, k2_u, k2_v, k3_u, k3_v, k4_u, k4_v, dt)
        return cast(tuple[Grid2D, Grid2D], result)


class PredictorCorrectorIntegrator(TimeIntegratorStrategy):
    def advance_time(
        self,
        u: Grid2D,
        v: Grid2D,
        p_prev: Grid2D,
        dt: float,
        method: SpatialDiscretizationStrategy,
        dx: float,
        dy: float,
        nu: float,
    ) -> tuple[Grid2D, Grid2D]:
        """
        Advance velocity fields in time using predictor-corrector method without pressure gradient.
        This method improves accuracy over a simple forward Euler scheme by taking an average of two estimates of the time derivative, effectively achieving second-order accuracy in time.

        Args:
            u (Grid2D): Current x-velocity field.
            v (Grid2D): Current y-velocity field.
            p_prev (Grid2D): Previous pressure field (unused in predictor step).
            dt (float): Time step size.
            method (SpatialDiscretizationStrategy): Spatial discretization method.
            dx (float): Grid spacing in x-direction.
            dy (float): Grid spacing in y-direction.
            nu (float): Kinematic viscosity.

        Returns:
            tuple[Grid2D, Grid2D]: Intermediate velocity fields (u, v).
        """
        # Predictor step
        du_dt1, dv_dt1 = method(u, v, p_prev, dx, dy, nu)
        u_star = u + dt * du_dt1
        v_star = v + dt * dv_dt1

        # Corrector step
        du_dt2, dv_dt2 = method(u_star, v_star, p_prev, dx, dy, nu)

        # Average predictor and corrector
        nx, ny = u.shape
        u_new = np.zeros_like(u)
        v_new = np.zeros_like(v)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_new[i, j] = u[i, j] + 0.5 * dt * (du_dt1[i, j] + du_dt2[i, j])
                v_new[i, j] = v[i, j] + 0.5 * dt * (dv_dt1[i, j] + dv_dt2[i, j])

        return cast(tuple[Grid2D, Grid2D], (u_new, v_new))
