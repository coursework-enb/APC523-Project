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
    Numba-optimized implementation.
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
    Numba-optimized implementation.
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


@njit(parallel=True)
def semi_implicit_step(
    u: Grid2D,
    v: Grid2D,
    adv_u: Grid2D,
    adv_v: Grid2D,
    dt: float,
    nu: float,
    dx: float,
    dy: float,
) -> tuple[Grid2D, Grid2D]:
    """
    Perform a semi-implicit time step for velocity fields, treating advection explicitly
    and diffusion implicitly.
    Numba-optimized implementation.
    """
    nx, ny = u.shape
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    denom = 1.0 + 2.0 * dt * nu * (1.0 / dx**2 + 1.0 / dy**2)

    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            # Implicit diffusion for u
            diff_u = nu * (
                (u[i, j + 1] + u[i, j - 1]) / dx**2 + (u[i + 1, j] + u[i - 1, j]) / dy**2
            )
            u_new[i, j] = (u[i, j] + dt * adv_u[i, j] + dt * diff_u) / denom

            # Implicit diffusion for v
            diff_v = nu * (
                (v[i, j + 1] + v[i, j - 1]) / dx**2 + (v[i + 1, j] + v[i - 1, j]) / dy**2
            )
            v_new[i, j] = (v[i, j] + dt * adv_v[i, j] + dt * diff_v) / denom

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


class SemiImplicitIntegrator(TimeIntegratorStrategy):
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
        Advance velocity fields in time using a semi-implicit method without pressure gradient.
        Advection terms are treated explicitly, while diffusion terms are treated implicitly
        to relax time step restrictions due to viscosity.

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
        du_dt, dv_dt = method(u, v, p_prev, dx, dy, nu)
        result = semi_implicit_step(u, v, du_dt, dv_dt, dt, nu, dx, dy)
        return cast(tuple[Grid2D, Grid2D], result)
