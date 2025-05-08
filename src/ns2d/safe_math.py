"""
This module contains mathematical functions that leverage Numba and are designed for safety,
in particular to mitigate overflow during quadratic terms in velocity:

1. u^2 + v^2
2. sum(u^2 + v^2)
3. sqrt(u^2 + v^2)
4. max(sqrt(u^2 + v^2))

Of course it cannot completely prevent overflow: choosing a time step that is too large for a
given velocity and grid spacing can lead to instabilities and exponential growth, which will
lead to overflows.
"""

import numpy as np
from numba import njit

from .utils import Grid2D


@njit
def calculate_squared_speeds(u: Grid2D, v: Grid2D) -> Grid2D:
    """
    Safe computation of element-wise u^2 + v^2.
    This calculates the square of the local speed at each point.
    """
    result = np.empty(u.shape, dtype=np.float64)

    for i in range(u.size):
        u_val = u.flat[i]
        v_val = v.flat[i]

        result.flat[i] = float(u_val) ** 2 + float(v_val) ** 2

    return result


@njit
def calculate_aggr_sq_speeds(u: Grid2D, v: Grid2D) -> float:
    """
    Safe computation of the sum of u^2 + v^2.
    This aggregates the squared speeds over the domain.
    """
    total_sum = 0.0

    for i in range(u.size):
        u_val = u.flat[i]
        v_val = v.flat[i]

        total_sum += float(u_val) ** 2 + float(v_val) ** 2
    return total_sum


@njit
def compute_velocity_magnitude(u: Grid2D, v: Grid2D) -> Grid2D:
    """
    Safe computation of element-wise sqrt(u^2 + v^2).
    This calculates the local speed (magnitude of velocity) at each point.
    """
    result = np.empty(u.shape, dtype=np.float64)

    for i in range(u.size):
        u_val = u.flat[i]
        v_val = v.flat[i]

        squared_magnitude = float(u_val) ** 2 + float(v_val) ** 2

        result.flat[i] = squared_magnitude**0.5

    return result


@njit
def calculate_max_velocity_magn(u: Grid2D, v: Grid2D) -> float:
    """
    Safe calculation of the maximum square velocity.
    """
    u_val_init = u.flat[0]
    v_val_init = v.flat[0]
    max_speed: float = (float(u_val_init) ** 2 + float(v_val_init) ** 2) ** 0.5

    for i in range(1, u.size):
        u_val = u.flat[i]
        v_val = v.flat[i]

        current_speed = (float(u_val) ** 2 + float(v_val) ** 2) ** 0.5

        if current_speed > max_speed:
            max_speed = current_speed

    return max_speed
