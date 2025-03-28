import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ns2d import (
    NavierStokesSolver2D,
    SpatialDiscretizationStrategy,
    TimeIntegratorStrategy
)
from numpy import zeros, allclose

# Create concrete implementations for testing
class TestDiscretizer(SpatialDiscretizationStrategy):
    def __call__(self, u, v, p, dx, dy, nu):
        return u.copy(), v.copy()  # dummy implementation

class TestIntegrator(TimeIntegratorStrategy):
    def advance_time(self, u, v, p_prev, dt, method, dx, dy, nu):
        return u.copy(), v.copy()  # dummy implementation

# Testing base initialization
class TestSolver(NavierStokesSolver2D):
    def initialize_fields(self):
        pass

    def solve_poisson(self):
        pass

    def update_velocity(self):
        pass

    def compute_vorticity(self):
        return zeros(self.nx, self.ny)

    def validate(self, benchmark: str) -> None:
        pass

class TestSolverDimensions(unittest.TestCase):
    def setUp(self):
        self.nx, self.ny = 10, 20
        self.solver = TestSolver(
            nx=self.nx,
            ny=self.ny,
            dx=0.1,
            dy=0.1,
            dt=0.01,
            nu=0.1,
            integrator=TestIntegrator(),
            discrete_navier_stokes=TestDiscretizer()
        )

    def test_grid_dimensions(self):
        """Verify grid dimensions match initialization parameters"""
        self.assertEqual(self.solver.u.shape, (self.nx, self.ny))
        self.assertEqual(self.solver.v.shape, (self.nx, self.ny))
        self.assertEqual(self.solver.p.shape, (self.nx, self.ny))

    def test_initial_values(self):
        """Check all fields initialize to zero"""
        self.assertTrue(allclose(self.solver.u, 0.0))
        self.assertTrue(allclose(self.solver.v, 0.0))
        self.assertTrue(allclose(self.solver.p, 0.0))

if __name__ == '__main__':
    unittest.main()
