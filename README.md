# Installation

For development purposes

- Clone the repository
  ```
  git clone git@github.com:coursework-enb/APC523-Project.git
  cd APC523-Project
  ```
- Reproduce the Python environment
  ```
  uv venv .venv --python 3.12.9
  source .venv/bin/activate
  uv pip install -r requirements.txt
  ```
  Notes:
  - The package `ns2d` is already installed in editable mode
  - To install `mpi4py`, make sure you have the `mpicc` binary (from OpenMPI or MPICH) and that it is in your system's `PATH` (you can find it using `find /usr -name mpicc`)
- Set up Git hooks
  ```
  pre-commit install
  ```
- To use Prettier outside of pre-commit hooks, you will need to install it locally with:
  ```
  npm install --save-dev prettier@4.0.0-alpha.8 prettier-plugin-latex
  ```

# Programming Tips

- Numerical tools:
  - Use `scipy.fft` or pyFFTW for speed-critical FFTs
  - Use PyAMG for performance multigrid
- Profiling:
  - Use cProfile or Py-Spy during development to identify bottlenecks
  - Use `tqdm` to monitor progress
  - For memory-intensive simulations, use `memory_profiler`
  - For multi-threaded code use Scalene

* Performance:
  - Use Numba to optimize your code via JIT compilation, and multithreading
  - Use `mpi4py` for parallel computing (between multiple processors or nodes)
  - Use JAX to accelerate computations on GPUs
* Visualization:
  - Use Matplotlib, Seaborn and Plot-utils for static plots
  - Use Plotly for interactive 2D/3D visualizations
