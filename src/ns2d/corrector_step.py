from ns2d import NavierStokesSolver2D

# -----------------------
# Numba functions
# -----------------------


# -----------------------
# Solvers
# -----------------------


class JacobiSolver(NavierStokesSolver2D): ...


class GaussSeidelSolver(NavierStokesSolver2D): ...


class ConjugateGradientSolver(NavierStokesSolver2D): ...
