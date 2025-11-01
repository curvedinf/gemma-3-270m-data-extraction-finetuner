"""Task implementations used by Fabric collections."""

TRAIN_IMPORT_ERROR = None
try:
    from . import train  # type: ignore
except NotImplementedError as exc:  # GPU-only dependency
    TRAIN_IMPORT_ERROR = exc
    train = None  # type: ignore
except Exception:
    # Surface other import errors instead of hiding them
    raise
else:
    TRAIN_IMPORT_ERROR = None

from . import dataset, env, evaluate, ops, package

__all__ = ["dataset", "env", "evaluate", "ops", "package"]
if train is not None:  # type: ignore
    __all__.append("train")
