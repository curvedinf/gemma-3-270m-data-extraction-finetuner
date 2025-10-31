"""Environment management tasks."""

from pathlib import Path

from fabric import task

DEFAULT_VENV = Path(".venv")
REQUIREMENTS_FILE = Path("requirements.txt")


def _python_bin(venv: Path) -> Path:
    return venv / "bin" / "python"


@task(help={"python": "Path to the Python executable for bootstrapping the venv."})
def bootstrap(c, python="python3"):
    """
    Create (or refresh) the project virtual environment and install dependencies.

    Uses `.venv` relative to the repo root. Safe to re-run; installs from
    `requirements.txt`.
    """
    if not DEFAULT_VENV.exists():
        c.run(f"{python} -m venv {DEFAULT_VENV}", pty=True)

    python_bin = _python_bin(DEFAULT_VENV)
    c.run(f"{python_bin} -m pip install --upgrade pip", pty=True)
    if REQUIREMENTS_FILE.exists():
        c.run(f"{python_bin} -m pip install -r {REQUIREMENTS_FILE}", pty=True)
    else:
        print("requirements.txt not found; skipping dependency installation.")


@task
def lock(c):
    """
    Compile dependency lockfile using `uv pip compile` or `pip-tools` if available.

    This is a placeholder that should be expanded once the dependency management
    strategy (uv vs pip-tools) is finalized.
    """
    print("Locking dependencies not yet implemented. Decide on uv/pip-tools flow.")


@task
def check(c):
    """
    Verify the virtual environment and Python tooling health.

    Intended to run quick diagnostics (python --version, pip list). Extend with
    additional checks as the project matures.
    """
    python_bin = _python_bin(DEFAULT_VENV)
    if not python_bin.exists():
        raise RuntimeError("Virtual environment missing. Run `fab env.bootstrap` first.")

    c.run(f"{python_bin} --version", pty=True)
    c.run(f"{python_bin} -m pip --version", pty=True)
