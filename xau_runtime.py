"""
xau_runtime.py — keep project modules in sync with the files on disk.

Streamlit Cloud re-runs app.py after a git push but can keep previously
imported modules (xau_config, xau_*.py) cached in memory, so new code meets
old settings and fails with AttributeError until the app is rebooted.

ensure_fresh() records each module's file mtime on first import; on later
runs, if any file is newer than its loaded copy, every listed module is
reloaded in the given (dependency) order. Reloading updates the module
objects in place, so existing `import xau_config as cfg` references see the
new values.
"""
import importlib
import os
from types import ModuleType
from typing import List, Sequence

_ATTR = "__xau_loaded_mtime__"


def _mtime(mod: ModuleType) -> float:
    try:
        return os.path.getmtime(mod.__file__)
    except (OSError, TypeError, AttributeError):
        return 0.0


def stale_modules(modules: Sequence[ModuleType]) -> List[str]:
    out = []
    for m in modules:
        loaded = getattr(m, _ATTR, None)
        if loaded is None:
            setattr(m, _ATTR, _mtime(m))       # first sighting: record, not stale
        elif _mtime(m) > loaded:
            out.append(m.__name__)
    return out


def ensure_fresh(modules: Sequence[ModuleType]) -> List[str]:
    """Reload all modules (in order) if any changed on disk. Returns the names
    that triggered the reload (empty list if nothing changed)."""
    stale = stale_modules(modules)
    if stale:
        for m in modules:
            importlib.reload(m)
            setattr(m, _ATTR, _mtime(m))
    return stale
