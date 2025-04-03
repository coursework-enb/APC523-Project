# Installation

For development purposes

* Clone the repository
  ```
  git clone git@github.com:coursework-enb/APC523-Project.git
  cd APC523-Project
  ```
* Reproduce the Python environment
  ```
  uv venv .venv --python 3.12.9
  source .venv/bin/activate
  uv pip install -r requirements.txt
  ```
  Notes:
  + The package `ns2d` is already installed in editable mode
  + To install `mpi4py`, make sure you have the `mpicc` binary (from OpenMPI or MPICH) and that it is in your system's `PATH`
* Set up Git hooks
  ```
  pre-commit install
  ```
* To use Prettier outside of pre-commit hooks, you will need to install it locally with:
  ```
  npm install --save-dev prettier@4.0.0-alpha.8 prettier-plugin-latex
  ```
