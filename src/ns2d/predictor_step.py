# TODO: Implement actual Crank-Nicolson, which is second order in time, for the
# semi-implicit scheme (instead of backward Euler)

from typing import cast

import numpy as np
from numba import njit, prange

from ns2d import SpatialDiscretizationStrategy, TimeIntegratorStrategy

from .boundaries import apply_velocity_bc
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

    # Interior points
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

    # Interior points
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
def jacobi_diffusion_solver(
    velocity: Grid2D, nu: float, dt: float, dx: float, dy: float, max_iter: int = 20
) -> Grid2D:
    """
    Jacobi solver for implicit diffusion terms, approximating the solution to the equation (I - dt * ν/2 * ∇²) u* = RHS
    Numba-optimized implementation.
    """
    nx, ny = velocity.shape
    velocity_new = velocity.copy()
    alpha = nu * dt
    dx2 = dx**2
    dy2 = dy**2
    denominator = 1 + 2 * alpha * (1 / dx2 + 1 / dy2)

    for _ in range(max_iter):
        velocity_old = velocity_new.copy()

        # Interior points
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                # Using central differences for the diffusion operator
                diffusion_term = (
                    velocity_old[i + 1, j] + velocity_old[i - 1, j]
                ) / dy2 + (velocity_old[i, j + 1] + velocity_old[i, j - 1]) / dx2
                velocity_new[i, j] = (
                    velocity[i, j] + alpha * diffusion_term
                ) / denominator

    return velocity_new


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
        bc_case: int,
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
        du_dt, dv_dt = method(u, v, p_prev, dx, dy, nu, bc_case)
        u_new, v_new = euler_step(u, v, du_dt, dv_dt, dt)

        # Boundaries
        result = apply_velocity_bc(u_new, v_new, bc_case)

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
        bc_case: int,
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
        k1_u, k1_v = method(u, v, p_prev, dx, dy, nu, bc_case)
        u1 = u + 0.5 * dt * k1_u
        v1 = v + 0.5 * dt * k1_v

        # Stage 2
        k2_u, k2_v = method(u1, v1, p_prev, dx, dy, nu, bc_case)
        u2 = u + 0.5 * dt * k2_u
        v2 = v + 0.5 * dt * k2_v

        # Stage 3
        k3_u, k3_v = method(u2, v2, p_prev, dx, dy, nu, bc_case)
        u3 = u + dt * k3_u
        v3 = v + dt * k3_v

        # Stage 4
        k4_u, k4_v = method(u3, v3, p_prev, dx, dy, nu, bc_case)

        # Final update (interior points)
        u_new, v_new = rk4_step(u, v, k1_u, k1_v, k2_u, k2_v, k3_u, k3_v, k4_u, k4_v, dt)

        # Boundaries
        result = apply_velocity_bc(u_new, v_new, bc_case)

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
        bc_case: int,
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
        du_dt1, dv_dt1 = method(u, v, p_prev, dx, dy, nu, bc_case)
        u_star = u + dt * du_dt1
        v_star = v + dt * dv_dt1

        # Corrector step
        du_dt2, dv_dt2 = method(u_star, v_star, p_prev, dx, dy, nu, bc_case)

        # Average predictor and corrector
        u_new = u.copy()
        v_new = v.copy()
        u_new[1:-1, 1:-1] += 0.5 * dt * (du_dt1[1:-1, 1:-1] + du_dt2[1:-1, 1:-1])
        v_new[1:-1, 1:-1] += 0.5 * dt * (dv_dt1[1:-1, 1:-1] + dv_dt2[1:-1, 1:-1])

        result = apply_velocity_bc(u_new, v_new, bc_case)

        return cast(tuple[Grid2D, Grid2D], result)


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
        bc_case: int,
    ) -> tuple[Grid2D, Grid2D]:
        """
        Advance velocity fields using semi-implicit scheme:
        - Explicit treatment of advection/pressure terms
        - Jacobi-iterated implicit treatment of diffusion

        Args:
            u: Current x-velocity field
            v: Current y-velocity field
            p_prev: Previous pressure field
            dt: Time step size
            method: Spatial discretization strategy
            dx: Grid spacing in x-direction
            dy: Grid spacing in y-direction
            nu: Kinematic viscosity
            bc_case: Boundary condition case identifier

        Returns:
            Updated velocity fields (u, v) after time step
        """
        du_dt_explicit, dv_dt_explicit = method(u, v, p_prev, dx, dy, nu, bc_case)

        u_star = u + dt * du_dt_explicit
        v_star = v + dt * dv_dt_explicit

        u_new = jacobi_diffusion_solver(u_star, nu, dt, dx, dy)
        v_new = jacobi_diffusion_solver(v_star, nu, dt, dx, dy)

        result = apply_velocity_bc(u_new, v_new, bc_case)

        return cast(tuple[Grid2D, Grid2D], result)
