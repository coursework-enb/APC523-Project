"""
This script implements two cases of boundary conditions, relevant to our benchmarks.
We use numba so that functions can be called from another numba function (or not).

TODO: Bring other BC designs in vorticity, stream, etc. from core.py here to centralize BC choices.
"""

from numba import njit

from .utils import Grid2D


@njit()
def apply_velocity_bc(u: Grid2D, v: Grid2D, bc_case: int) -> tuple[Grid2D, Grid2D]:
    """
    Apply boundary conditions to the velocity fields based on the specified case.

    Args:
        u (Grid2D): 2D grid of x-component of velocity
        v (Grid2D): 2D grid of y-component of velocity
        bc_case (int): Integer indicating the benchmark case (1 for Taylor-Green Vortex, 2 for Lid-Driven Cavity)

    Returns:
        tuple[Grid2D, Grid2D]: Updated u and v grids after applying velocity boundary conditions
    """
    if bc_case == 1:
        # TGV-type: Periodic boundary conditions

        u[0, :] = u[-2, :]  # Periodic in x-direction (left boundary)
        u[-1, :] = u[1, :]  # Periodic in x-direction (right boundary)
        u[:, 0] = u[:, -2]  # Periodic in y-direction (bottom boundary)
        u[:, -1] = u[:, 1]  # Periodic in y-direction (top boundary)

        v[0, :] = v[-2, :]
        v[-1, :] = v[1, :]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]

    elif bc_case == 2:
        # LDC-type: No-slip conditions on all walls except the top wall
        # Top wall has a tangential velocity (u=1, v=0), others have u=v=0

        # Bottom wall (y=0): No-slip (u=0, v=0)
        u[:, 0] = 0.0
        v[:, 0] = 0.0

        # Top wall (y=ny-1): Lid moving to the right (u=1, v=0)
        u[:, -1] = 1.0
        v[:, -1] = 0.0

        # Left wall (x=0): No-slip (u=0, v=0)
        u[0, :] = 0.0
        v[0, :] = 0.0

        # Right wall (x=nx-1): No-slip (u=0, v=0)
        u[-1, :] = 0.0
        v[-1, :] = 0.0

    else:
        raise ValueError(f"Unknown boundary condition case: {bc_case}")

    return u, v


@njit()
def apply_pressure_bc(p: Grid2D, bc_case: int) -> Grid2D:
    """
    Apply boundary conditions to the pressure field based on the specified case.

    Args:
        p (Grid2D): 2D grid of pressure
        bc_case (int): Integer indicating the benchmark case (1 for Taylor-Green Vortex, 2 for Lid-Driven Cavity)

    Returns:
        Grid2D: Updated p grid after applying pressure boundary conditions
    """
    if bc_case == 1:
        # Periodic boundary conditions
        p[0, :] = p[-2, :]
        p[-1, :] = p[1, :]
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]

    elif bc_case == 2:
        # Neumann conditions (dp/dn=0) for pressure
        p[:, 0] = p[:, 1]  # Bottom wall
        p[:, -1] = p[:, -2]  # Top wall
        p[0, :] = p[1, :]  # Left wall
        p[-1, :] = p[-2, :]  # Right wall

    else:
        raise ValueError(f"Unknown boundary condition case: {bc_case}")

    return p


@njit()
def apply_bc_rhs(
    du_dt: Grid2D,
    dv_dt: Grid2D,
    bc_case: int,
) -> tuple[Grid2D, Grid2D]:
    """
    Apply boundary conditions to the RHS time derivatives for momentum equations.

    :return: Updated time derivatives (du_dt, dv_dt) with BC applied.
    """

    if bc_case == 1:
        # Mirror the derivatives at boundaries to enforce periodicity
        du_dt[0, :] = du_dt[-2, :]
        du_dt[-1, :] = du_dt[1, :]
        du_dt[:, 0] = du_dt[:, -2]
        du_dt[:, -1] = du_dt[:, 1]

        dv_dt[0, :] = dv_dt[-2, :]
        dv_dt[-1, :] = dv_dt[1, :]
        dv_dt[:, 0] = dv_dt[:, -2]
        dv_dt[:, -1] = dv_dt[:, 1]

    elif bc_case == 2:
        # Ensure that time derivatives do not alter fixed boundary values of velocity
        du_dt[:, 0] = 0.0
        du_dt[:, -1] = 0.0
        du_dt[0, :] = 0.0
        du_dt[-1, :] = 0.0

        dv_dt[:, 0] = 0.0
        dv_dt[:, -1] = 0.0
        dv_dt[0, :] = 0.0
        dv_dt[-1, :] = 0.0

    return du_dt, dv_dt
