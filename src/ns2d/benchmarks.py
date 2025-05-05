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
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * decay
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * decay
    p = -0.25 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) * decay**2
    return u, v, p


def initialize_for_benchmark(
    benchmark: str, nx: int, ny: int
) -> tuple[Grid2D, Grid2D, Grid2D, Grid2D, Grid2D, int]:
    """
    Initialize velocity and pressure fields based on the specified benchmark problem.
    Note: ensure correct BC are applied for both (periodic vs tangential velocity on top)

    Args:
        benchmark: The benchmark problem to initialize for ('Taylor-Green Vortex' or 'Lid-Driven Cavity')
    """
    if benchmark == "Taylor-Green Vortex":
        case = 1

        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)

        u_init, v_init, p_init = _taylor_green_analytical_solution(X, Y, 0, 1)

    elif benchmark == "Lid-Driven Cavity":
        case = 2

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Zero velocity and pressure
        u_init = np.zeros_like(X)
        v_init = np.zeros_like(Y)
        p_init = np.zeros_like(X)

        # Top wall velocity u=(1,0)
        u_init[-1, :] = 1.0

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return X, Y, u_init, v_init, p_init, case


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
    verbose: bool,
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

        if verbose:
            print(f"Taylor-Green Vortex Validation at t={current_time}s:")
            print(f"Simulated Kinetic Energy: {ke_simulated}")
            print(f"Analytical Kinetic Energy: {ke_analytical}")
            print(f"Absolute Error in Kinetic Energy: {error_ke}")

        return error_ke

    elif benchmark == "Lid-Driven Cavity":
        assert current_time == 2.5 and reference_min_stream == -0.061076605

        min_stream_func = float(np.min(stream_func))
        error_stream: float = abs(min_stream_func - reference_min_stream)

        if verbose:
            print("Lid-Driven Cavity Validation at t=2.5s:")
            print(f"Simulated Minimum Stream Function: {min_stream_func}")
            print(f"Reference Minimum Stream Function: {reference_min_stream}")
            print(f"Absolute Error in Stream Function Minimum: {error_stream}")

        return error_stream

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
