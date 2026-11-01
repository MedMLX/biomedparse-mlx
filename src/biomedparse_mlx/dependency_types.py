"""Admission boundary for incompletely annotated dependency callables."""

from __future__ import annotations


def callable_member(owner: object, name: str) -> object:
    """Resolve one real external callable before applying its source-qualified signature."""
    value: object = getattr(owner, name)
    if not callable(value):
        raise TypeError(f"Dependency attribute {name} must be callable")
    return value
