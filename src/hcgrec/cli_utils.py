from typing import Any


_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def coerce_bool_arg(value: Any, arg_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"Invalid boolean value for {arg_name}: {value!r} ({type(value).__name__})")


def format_typed_value(value: Any) -> str:
    return f"{value!r} ({type(value).__name__})"
