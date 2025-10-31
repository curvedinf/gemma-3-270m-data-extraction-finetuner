"""Environment management tasks."""

from pathlib import Path

from fabric import task

DEFAULT_VENV = Path(".venv")
REQUIREMENTS_FILE = Path("requirements.txt")


def _python_bin(venv: Path) -> Path:
    return venv / "bin" / "python"


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
        raise RuntimeError("Virtual environment missing. Run `./scripts/setup_env.sh` first.")

    c.run(f"{python_bin} --version", pty=True)
    c.run(f"{python_bin} -m pip --version", pty=True)
