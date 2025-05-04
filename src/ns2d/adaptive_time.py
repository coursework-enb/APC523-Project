import numpy as np

from .utils import Grid2D


def adapt_time_step(
    u_prev: Grid2D,
    v_prev: Grid2D,
    u_curr: Grid2D,
    v_curr: Grid2D,
    dt: float,
    min_dt: float,
    max_dt: float,
    tol: float,
) -> tuple[float, bool]:
    """
    Adapt the time step based on the maximum relative change in velocity fields.

    Args:
        u_prev, v_prev: Previous velocity components (2D arrays)
        u_curr, v_curr: Current velocity components after a tentative step (2D arrays)
        dt: Current time step
        min_dt: Minimum allowed time step
        max_dt: Maximum allowed time step
        tol: Tolerance for accepting the step (default: 1e-1)

    Returns:
        tuple: (new time step, whether the step is accepted)
    """
    eps_u = np.max(np.abs(u_curr - u_prev) / (np.abs(u_prev) + 1e-10))
    eps_v = np.max(np.abs(v_curr - v_prev) / (np.abs(v_prev) + 1e-10))
    eps = max(eps_u, eps_v)

    accept = eps <= tol
    if accept:
        dt_new = min(max_dt, dt * 2.0)  # double the time step if accepted
    else:
        dt_new = max(min_dt, dt * 0.5)  # halve the time step if rejected

    return dt_new, bool(accept)
