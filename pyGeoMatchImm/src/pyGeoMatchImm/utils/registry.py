# Registry for components
# This registry is used to keep track of various components in the system

_REG = {}

def register(key):
    def deco(cls_or_fn):
        if key in _REG:
            raise KeyError(f"Duplicate registry key: {key}")
        _REG[key] = cls_or_fn
        return cls_or_fn
    return deco

def get(key):
    if key not in _REG:
        raise KeyError(f"Unknown component: {key}\nAvailable: {list(_REG)}")
    return _REG[key]
