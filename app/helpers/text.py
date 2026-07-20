"""Small text utilities shared across services."""

from typing import Iterable


def contains_any(text: str, terms: Iterable[str]) -> bool:
    """Case-insensitive: does `text` contain any of `terms`?"""
    lowered = text.lower()
    return any(term in lowered for term in terms)
