# Projection Method

We separate the computation into spatial discretization, time advancement of velocity (predictor), and pressure correction (corrector) to handle the incompressibility constraint.
This method decouple the velocity and pressure solutions, solving the momentum equations first to get a tentative velocity (ignoring incompressibility initially), then correcting it by projecting the velocity field onto a divergence-free space via a pressure Poisson equation.

1. Spatial discretization strategies for the right-hand side of the momentum equations. This step involves discretizing the spatial terms (RHS) of the momentum equations, which includes the advection (convection), diffusion, and pressure gradient terms.
2. Then we handle the time derivative (left-hand side) of the momentum equations by simply taking the spatially discretized equations and advancing the velocity fields (u and v) in time without considering the continuity constraint ($\nabla u=0$). This is the predictor step, computing a tentative velocity field based on the current state, which performs a transport phase that accounts for advection and diffusion without satisfying incompressibility.
3. Finally a new pressure field is computed to enforce incompressibility by solving a Poisson equation for pressure (relating pressure at any point to the entire velocity field). Then the velocity field is corrected to satisfy the continuity equation.

This method was prefered to a coupled (monolithic) method that solves the velocity and pressure fields simultaneously in a fully coupled system at each time step, since it decreases computational cost, although it cannot achieve higher-order temporal accuracy as easily.
This projection method also makes it easier to analyze global error and convergence rate, and separate the work in different tasks.
