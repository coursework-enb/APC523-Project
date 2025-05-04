from typing import cast

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

import .core as ns2dcore


def _get_stencil_coefficients(order: int) -> tuple[NDArray[np.float64], int]:
    """
    Return stencil coefficients and offset for central finite difference of given order.
    Coefficients are for first derivative approximation.

    Args:
        order (int): Order of accuracy for finite difference (must be even, e.g., 2, 4, 6).

    Returns:
        tuple[np.ndarray, int]: Coefficients array and offset (half the stencil width).

    Raises:
        ValueError: If order is not even or unsupported.
    """
    if order % 2 != 0:
        raise ValueError(f"Order must be even, got {order}")

    half_width = order // 2
    if order == 2:
        coeffs = np.array([-0.5, 0.5])
        denom = 1.0
    elif order == 4:
        coeffs = np.array([1 / 12, -2 / 3, 2 / 3, -1 / 12])
        denom = 1.0
    elif order == 6:
        coeffs = np.array([-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60])
        denom = 1.0
    else:
        raise ValueError(
            f"Unsupported order of accuracy: {order}. Currently supported: 2, 4, 6."
        )

    return coeffs / denom, half_width


@njit(parallel=True)
def _compute_vorticity_central(
    u: ns2dcore.Grid2D, v: ns2dcore.Grid2D, dx: float, dy: float, order: int = 2
) -> ns2dcore.Grid2D:
    """
    Compute vorticity for interior points using central difference of specified order.

    Args:
        u (ns2dcore.Grid2D): Velocity component in x-direction.
        v (ns2dcore.Grid2D): Velocity component in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        order (int): Order of accuracy for finite difference (even number, e.g., 2, 4, 6).

    Returns:
        ns2dcore.Grid2D: Vorticity field (only interior points computed).
    """
    nx, ny = u.shape
    vorticity = np.zeros_like(u)
    coeffs, offset = _get_stencil_coefficients(order)

    for i in prange(offset, nx - offset):
        for j in range(offset, ny - offset):
            dv_dx = 0.0
            du_dy = 0.0
            for k in range(len(coeffs)):
                dv_dx += coeffs[k] * v[i, j + k - offset]
                du_dy += coeffs[k] * u[i + k - offset, j]
            dv_dx /= dx
            du_dy /= dy
            vorticity[i, j] = dv_dx - du_dy

    return vorticity


@njit(parallel=True)
def finite_difference_vorticity(
    u: ns2dcore.Grid2D, v: ns2dcore.Grid2D, dx: float, dy: float, order: int
) -> ns2dcore.Grid2D:
    """
    Compute vorticity using central difference for interior points and one-sided differences
    for boundary points.

    Args:
        u (ns2dcore.Grid2D): Velocity component in x-direction.
        v (ns2dcore.Grid2D): Velocity component in y-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        order (int): Order of accuracy for finite difference in interior points (e.g., 2, 4, 6).

    Returns:
        ns2dcore.Grid2D: Computed vorticity field with boundary adjustments.
    """
    nx, ny = u.shape
    vorticity = _compute_vorticity_central(u, v, dx, dy, order)

    # Boundary points: one-sided differences
    # Left boundary (j=0)
    for i in prange(nx):
        j = 0
        dv_dx = (v[i, j + 1] - v[i, j]) / dx
        if i == 0:
            du_dy = (u[i + 1, j] - u[i, j]) / dy
        elif i == nx - 1:
            du_dy = (u[i, j] - u[i - 1, j]) / dy
        else:
            du_dy = (u[i + 1, j] - u[i - 1, j]) / (2 * dy)
        vorticity[i, j] = dv_dx - du_dy

    # Right boundary (j=ny-1)
    for i in prange(nx):
        j = ny - 1
        dv_dx = (v[i, j] - v[i, j - 1]) / dx
        if i == 0:
            du_dy = (u[i + 1, j] - u[i, j]) / dy
        elif i == nx - 1:
            du_dy = (u[i, j] - u[i - 1, j]) / dy
        else:
            du_dy = (u[i + 1, j] - u[i - 1, j]) / (2 * dy)
        vorticity[i, j] = dv_dx - du_dy

    # Bottom boundary (i=0)
    for j in prange(ny):
        i = 0
        du_dy = (u[i + 1, j] - u[i, j]) / dy
        if j == 0:
            dv_dx = (v[i, j + 1] - v[i, j]) / dx
        elif j == ny - 1:
            dv_dx = (v[i, j] - v[i, j - 1]) / dx
        else:
            dv_dx = (v[i, j + 1] - v[i, j - 1]) / (2 * dx)
        vorticity[i, j] = dv_dx - du_dy

    # Top boundary (i=nx-1)
    for j in prange(ny):
        i = nx - 1
        du_dy = (u[i, j] - u[i - 1, j]) / dy
        if j == 0:
            dv_dx = (v[i, j + 1] - v[i, j]) / dx
        elif j == ny - 1:
            dv_dx = (v[i, j] - v[i, j - 1]) / dx
        else:
            dv_dx = (v[i, j + 1] - v[i, j - 1]) / (2 * dx)
        vorticity[i, j] = dv_dx - du_dy

    return cast(ns2dcore.Grid2D, vorticity)
