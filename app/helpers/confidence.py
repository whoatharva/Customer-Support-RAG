"""Confidence scoring for retrieval answers.

Public API:
  compute_confidence(top_score, has_direct_data) -> float
"""


def compute_confidence(top_score: float, has_direct_data: bool) -> float:
    """Direct DB data boosts confidence; otherwise use the raw Qdrant similarity."""
    if has_direct_data:
        return round(min(0.5 * top_score + 0.5, 1.0), 2)
    return round(min(top_score, 1.0), 2)
