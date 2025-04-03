# Installation

For development purposes

* Clone the repository
  ```
  git clone git@github.com:coursework-enb/APC523-Project.git
  cd APC523-Project
  ```
* Reproduce the python enviroment
  ```
  uv venv .venv --python 3.12.9
  source .venv/bin/activate
  uv pip install -r requirements.txt
  uv pip install -e .
  ```
* Set up Git hooks
  ```
  pre-commit install
  ```
* To use Prettier outside of pre-commit hooks, you will need to install it locally with:
  ```
  npm install --save-dev prettier@4.0.0-alpha.8 prettier-plugin-latex
  ```
