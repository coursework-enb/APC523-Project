from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import float64, sum, zeros

from .benchmarks import validate_against_benchmark
from .utils import Grid2D
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

    def compute_kinetic_energy(self) -> float:
        """
        Compute the kinetic energy of the velocity field, scaled by grid cell area.

        :return: Kinetic energy value
        """
        cell_area = self.dx * self.dy
        ke: float = 0.5 * cell_area * sum(self.u**2 + self.v**2)
        return ke

    def compute_vorticity(self, order: int = 2) -> Grid2D:
        """Compute vorticity for validation purposes.

        :return: A 2D list representing the vorticity field.
        """
        result: Grid2D = finite_difference_vorticity(
            self.u, self.v, self.dx, self.dy, order=order
        )
        return result

    def solve_stream_function(self) -> Grid2D:
        """
        Compute the stream function by solving the Poisson equation -∇²ψ = ω using the class's Poisson solver.
        Vorticity is used as the source term.

        Returns:
            A 2D array representing the stream function field
        """
        vorticity = self.compute_vorticity()

        temp_p = self.p.copy()
        self.p = vorticity
        self.solve_poisson()
        stream_func = self.p.copy()

        # Restore original pressure field
        # TODO: revise solve_poisson such that we can input/output the pressure without mutation (safer)
        self.p = temp_p

        # No-flow boundaries
        stream_func[:, 0] = 0  # Bottom wall
        stream_func[:, -1] = 0  # Top wall
        stream_func[0, :] = 0  # Left wall
        stream_func[-1, :] = 0  # Right wall

        return stream_func

    @abstractmethod
    def validate(self, benchmark: str, current_time: float) -> None:
        """
        This method compares the solver's output (e.g., kinetic energy for Taylor-Green Vortex or minimum stream function for Lid-Driven Cavity) against reference or analytical solutions for the specified benchmark problem.

        :param benchmark: The benchmark problem to validate against
        """
        _ = validate_against_benchmark(
            benchmark,
            self.dx,
            self.dy,
            self.nx,
            self.ny,
            self.nu,
            current_time,
            self.compute_kinetic_energy(),
            self.solve_stream_function(),
        )

    def run_simulation(
        self,
        num_steps: int,
        end_time: float = 2.5,
        initialize: bool = True,
        validate: bool = False,
        benchmark: str = "Lid-Driven Cavity",
    ) -> None:
        """Run the simulation for a specified number of time steps.

        :param num_steps: Number of time steps to simulate
        """
        if initialize:
            self.initialize_fields()

        current_time = 0.0
        for step in range(num_steps):
            current_time = step * self.dt
            if current_time > end_time:
                break

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
            self.validate(benchmark, current_time)
