import warnings
from typing import Any, Dict

_REGISTRY: Dict[str, Any] = {}


def register(key: str, *, warn_if_overwrite: bool = True):
    """
    Register an object under a string key.
    Safe to use with IPython autoreload: re-imports will overwrite.
    """
    def deco(obj):
        if key in _REGISTRY and warn_if_overwrite:
            warnings.warn(
                f"Overwriting existing registry key {key}: "
                f"{_REGISTRY[key]} -> {obj}",
                RuntimeWarning,
                stacklevel=2,
            )
        _REGISTRY[key] = obj
        return obj

    return deco

def get(key):
    if key not in _REGISTRY:
        raise KeyError(f"Unknown component: {key}\nAvailable: {list(_REGISTRY)}")
    return _REGISTRY[key]
