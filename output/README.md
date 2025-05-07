# Instructions

Run all simulations in the following fashion:

```
solver = JacobiSolver(
    nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nu=nu,
    integrator=EulerIntegrator(),
    discrete_navier_stokes=FiniteDifferenceDiscretizer(),
    fixed_dt=False
)
start_time = time.perf_counter()
time_values, cfl_values, error = solver.integrate(
    num_steps=None,
    end_time=0.5,
    benchmark="Taylor-Green Vortex"
)
compute_time = time.perf_counter() - start_time
vorticity = solver.compute_vorticity()
```

Of course with different solvers, checkout [here](https://github.com/coursework-enb/APC523-Project/blob/main/src/ns2d/__init__.py) everything we have.

Systematically run for both values of $\nu\in\{1\times10^{-3}, 1\times10^{-5}\}$ and get results for both benchmarks.
Run TGV up to a final time $T = 0.5$ seconds and LDC until a final time $T = 2.5$ seconds.

Each result should provide the error, the fraction of `cfl_values` over $1$ and the compute time.
For visualization pruposes one can also store the vorticity at final time.

# Simulations

Very importantly, store the results AND the log in order to keep all information about the specific parameters and models used for each simulation.
Store all of them in this folder `./output/`.

The goal is to explore stability and accuracy, particularly as a function of the chosen time step (which is why we should also run for `fixed_dt=True`) and the viscosity.
Most of the results on stability should be tested on different LHS strategies (not so much RHS or Poisson solver).

Note: In terms of time-stepping strategies, we have at least three.
