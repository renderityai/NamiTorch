from ..tensor import Tensor


def pin_batch(value, memo=None, active=None):
    memo = {} if memo is None else memo
    active = set() if active is None else active
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Tensor):
        result = value.pin_memory()
        memo[identity] = result
        return result
    if not isinstance(value, (dict, list, tuple)):
        return value
    if identity in active:
        raise ValueError("Cannot pin a batch containing cyclic containers.")
    active.add(identity)
    try:
        if isinstance(value, dict):
            result = {key: pin_batch(item, memo, active) for key, item in value.items()}
        else:
            items = [pin_batch(item, memo, active) for item in value]
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                result = type(value)(*items)
            else:
                result = tuple(items) if isinstance(value, tuple) else items
        memo[identity] = result
        return result
    finally:
        active.remove(identity)
