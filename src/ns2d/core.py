from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import float64, zeros

from ns2d import Grid2D

from .vorticity import finite_difference_vorticity


class SpatialDiscretizationStrategy(ABC):
    @abstractmethod
    def __call__(
        self, u: Grid2D, v: Grid2D, p: Grid2D, dx: float, dy: float, nu: float
    ) -> tuple[Grid2D, Grid2D]:
        """Discretize spatial derivatives for momentum equations"""
        pass


class TimeIntegratorStrategy(ABC):
    @abstractmethod
    def advance_time(
        self,
        u: Grid2D,
        v: Grid2D,
        p_prev: Grid2D,
        dt: float,
        method: SpatialDiscretizationStrategy,  # inject dependency
        dx: float,
        dy: float,
        nu: float,  # with necessary inputs for call
    ) -> tuple[Grid2D, Grid2D]:
        """Calculate intermediate velocities without pressure gradient"""
        pass


@dataclass
class NavierStokesSolver2D(ABC):
    """Context class that uses strategies for time integration, spatial discretization, and implements pressure solution"""

    nx: int
    ny: int
    dx: float
    dy: float
    dt: float
    nu: float
    integrator: TimeIntegratorStrategy
    discrete_navier_stokes: SpatialDiscretizationStrategy
    u: Grid2D = field(init=False)
    v: Grid2D = field(init=False)
    p: Grid2D = field(init=False)

    def __post_init__(self) -> None:
        self.u = zeros((self.nx, self.ny), dtype=float64)
        self.v = zeros((self.nx, self.ny), dtype=float64)
        self.p = zeros((self.nx, self.ny), dtype=float64)

    def set_time_integrator(self, integrator: TimeIntegratorStrategy) -> None:
        """Change the time integration strategy at runtime"""
        self.integrator = integrator

    def set_spatial_discretizer(self, discretizer: SpatialDiscretizationStrategy) -> None:
        """Change the spatial discretization strategy at runtime"""
        self.discrete_navier_stokes = discretizer

    @abstractmethod
    def initialize_fields(self) -> None:
        """Initialize velocity and pressure fields."""
        raise NotImplementedError("Initialization not implemented")

    @abstractmethod
    def solve_poisson(self) -> None:
        """Solve the pressure Poisson equation"""
        raise NotImplementedError("Poisson solver not implemented")

    @abstractmethod
    def update_velocity(self) -> None:
        """Update velocity fields based on the pressure correction from the Poisson solver."""
        raise NotImplementedError("Velocity update function not implemented")

    def compute_vorticity(self, order: int = 2) -> Grid2D:
        """Compute vorticity for validation purposes.

        :return: A 2D list representing the vorticity field.
        """
        result: Grid2D = finite_difference_vorticity(
            self.u, self.v, self.dx, self.dy, order=order
        )
        return result

    @abstractmethod
    def validate(self, benchmark: str) -> None:
        """Validate the solver against benchmark problems like Taylor-Green vortex or lid-driven cavity.

        :param benchmark: The benchmark problem to validate against
        """
        raise NotImplementedError("Validation function not implemented")

    def run_simulation(
        self,
        num_steps: int,
        initialize: bool = True,
        validate: bool = False,
        benchmark: str = "Taylor-Green Vortex",
    ) -> None:
        """Run the simulation for a specified number of time steps.

        :param num_steps: Number of time steps to simulate
        """
        if initialize:
            self.initialize_fields()

        for step in range(num_steps):
            self.u, self.v = self.integrator.advance_time(
                self.u,
                self.v,
                self.p,
                self.dt,
                self.discrete_navier_stokes,
                self.dx,
                self.dy,
                self.nu,
            )
            self.solve_poisson()
            self.update_velocity()

            if validate and step % 10 == 0:
                self.compute_vorticity()

        if validate:
            self.validate(benchmark)
