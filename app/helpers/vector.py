"""Vector math helpers for embedding blending.

Public API:
  blend_and_normalise(query_vector, hyde_vector, query_weight, hyde_weight) -> list[float]
"""


def blend_and_normalise(
    query_vector: list[float],
    hyde_vector: list[float],
    query_weight: float,
    hyde_weight: float,
) -> list[float]:
    """Weighted blend of two vectors, L2-normalised.

    Returns the (unnormalised) query_vector if the blend has zero magnitude.
    """
    blended = [query_weight * q + hyde_weight * h for q, h in zip(query_vector, hyde_vector)]
    magnitude = sum(x * x for x in blended) ** 0.5
    if magnitude == 0:
        return query_vector
    return [x / magnitude for x in blended]
