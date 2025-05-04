import numpy as np

from .utils import Grid2D


def _taylor_green_analytical_solution(
    x: Grid2D, y: Grid2D, t: float, nu: float
) -> tuple[Grid2D, Grid2D, Grid2D]:
    """
    Compute the analytical solution for the Taylor-Green Vortex problem.

    Args:
        x: 2D grid of x-coordinates
        y: 2D grid of y-coordinates
        t: Current time
        nu: Kinematic viscosity

    Returns:
        Tuple of u, v, and p components of the analytical solution
    """
    decay = np.exp(-2 * nu * t * np.pi**2)
    u = np.cos(x) * np.sin(y) * decay
    v = -np.cos(y) * np.sin(x) * decay
    p = 0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay**2
    return u, v, p


def initialize_for_benchmark(
    benchmark: str, nx: int, ny: int
) -> tuple[Grid2D, Grid2D, Grid2D, int]:
    """
    Initialize velocity and pressure fields based on the specified benchmark problem.
    Note: ensure correct BC are applied for both (periodic vs tangential velocity on top)

    Args:
        benchmark: The benchmark problem to initialize for ('Taylor-Green Vortex' or 'Lid-Driven Cavity')
    """
    if benchmark == "Taylor-Green Vortex":
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)

        u = np.cos(X) * np.sin(Y)
        v = -np.cos(Y) * np.sin(X)
        p = 0.25 * (np.cos(2 * X) + np.cos(2 * Y))
        case = 1

    elif benchmark == "Lid-Driven Cavity":
        u = np.zeros((nx, ny))
        v = np.zeros((nx, ny))
        p = np.zeros((nx, ny))

        # Set top wall velocity to u=1 (lid moving to the right)
        u[:, -1] = 1.0
        # No-slip conditions (u=v=0) on other walls are already set by zero initialization

        case = 2

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return u, v, p, case


def apply_boundary_conditions(
    u: Grid2D, v: Grid2D, p: Grid2D, bc_case: int
) -> tuple[Grid2D, Grid2D, Grid2D]:
    """
    Apply boundary conditions to the velocity and pressure fields based on the specified case.

    Args:
        u: 2D grid of x-component of velocity
        v: 2D grid of y-component of velocity
        p: 2D grid of pressure
        bc_case: Integer indicating the benchmark case (1 for Taylor-Green Vortex, 2 for Lid-Driven Cavity)

    Returns:
        Tuple of updated u, v, and p grids after applying boundary conditions
    """
    if bc_case == 1:
        # Taylor-Green Vortex: Periodic boundary conditions

        u[0, :] = u[-2, :]  # Periodic in x-direction (left boundary)
        u[-1, :] = u[1, :]  # Periodic in x-direction (right boundary)
        u[:, 0] = u[:, -2]  # Periodic in y-direction (bottom boundary)
        u[:, -1] = u[:, 1]  # Periodic in y-direction (top boundary)

        v[0, :] = v[-2, :]
        v[-1, :] = v[1, :]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]

        p[0, :] = p[-2, :]
        p[-1, :] = p[1, :]
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]

    elif bc_case == 2:
        # Lid-Driven Cavity: No-slip conditions on all walls except the top wall
        # Top wall has a tangential velocity (u=1, v=0), others have u=v=0
        # Pressure boundary conditions are handled via Neumann conditions (dp/dn=0)

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

        # Set boundary pressure equal to adjacent interior point
        p[:, 0] = p[:, 1]  # Bottom wall
        p[:, -1] = p[:, -2]  # Top wall
        p[0, :] = p[1, :]  # Left wall
        p[-1, :] = p[-2, :]  # Right wall

    else:
        raise ValueError(f"Unknown boundary condition case: {bc_case}")

    return u, v, p


def validate_against_benchmark(
    benchmark: str,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    nu: float,
    current_time: float,
    ke_simulated: float,
    stream_func: Grid2D,
    reference_min_stream: float = -0.061076605,
) -> float:
    """
    Validate the solver against benchmark problems like Taylor-Green Vortex or Lid-Driven Cavity.

    Args:
        benchmark: The benchmark problem to validate against ('Taylor-Green Vortex' or 'Lid-Driven Cavity')
        reference_min_stream: Reference value for the minimum stream function, default at the final time T = 2.5 seconds
    """
    if benchmark == "Taylor-Green Vortex":
        # Set up grid for analytical solution
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)

        u_analytical, v_analytical, p_analytical = _taylor_green_analytical_solution(
            X, Y, current_time, nu
        )

        # Compute kinetic energy for comparison
        ke_analytical = 0.5 * dx * dy * np.sum(u_analytical**2 + v_analytical**2)
        error_ke: float = abs(ke_simulated - ke_analytical)

        print(f"Taylor-Green Vortex Validation at t={current_time}s:")
        print(f"Simulated Kinetic Energy: {ke_simulated}")
        print(f"Analytical Kinetic Energy: {ke_analytical}")
        print(f"Absolute Error in Kinetic Energy: {error_ke}")

        return error_ke

    elif benchmark == "Lid-Driven Cavity":
        assert current_time == 2.5 and reference_min_stream == -0.061076605

        min_stream_func = float(np.min(stream_func))
        error_stream: float = abs(min_stream_func - reference_min_stream)

        print("Lid-Driven Cavity Validation at t=2.5s:")
        print(f"Simulated Minimum Stream Function: {min_stream_func}")
        print(f"Reference Minimum Stream Function: {reference_min_stream}")
        print(f"Absolute Error in Stream Function Minimum: {error_stream}")

        return error_stream

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
