import nox
from pathlib import Path

nox.options.sessions = ["lint", "typecheck", "test"]
python_versions = ["3.10", "3.11"]
package_dir = Path("src/ns-solver")
tests_dir = Path("src/tests")

@nox.session(python=python_versions)
def lint(session):
    """Run code style checks"""
    session.install("flake8", "black", "isort")
    session.run("black", "--check", str(package_dir))
    session.run("isort", "--check-only", str(package_dir))
    session.run("flake8", str(package_dir))

@nox.session(python=python_versions)
def typecheck(session):
    """Run static type checking"""
    session.install("mypy", "numpy")
    session.run("mypy", "--strict", "--exclude", "tests/", str(package_dir))

@nox.session(python=python_versions)
def test(session):
    """Run unit tests"""
    session.install("numpy", "pytest", ".", "-r", "requirements.txt")
    session.run("pytest", "-v", str(tests_dir))

@nox.session(python=python_versions)
def docs(session):
    """Build documentation"""
    session.install("sphinx", "sphinx-rtd-theme")
    session.chdir("docs")
    session.run("sphinx-build", "-b", "html", ".", "_build/html")
