from typing import cast

import numpy as np
from numba import njit, prange

from ns2d import NavierStokesSolver2D

from .utils import Grid2D


# -----------------------
# Numba functions
# -----------------------


# -----------------------
# Solvers
# -----------------------

class JacobiSolver(NavierStokesSolver2D):
    ...

class GaussSeidelSolver(NavierStokesSolver2D):
    ...

class ConjugateGradientSolver(NavierStokesSolver2D):
    ...
