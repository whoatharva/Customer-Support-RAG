from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, QueryResponse
from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)
_client: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, verify=False)
        logger.info("Qdrant client initialized: url=%s", settings.qdrant_url)
    return _client


def ensure_collection():
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection_name not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info("collection created: %s", settings.qdrant_collection_name)
    else:
        logger.info("collection already exists: %s", settings.qdrant_collection_name)


def upsert_points(points: list[PointStruct]):
    logger.debug("upserting %d points", len(points))
    try:
        get_qdrant().upsert(collection_name=settings.qdrant_collection_name, points=points)
    except Exception:
        logger.error("failed to upsert %d points", len(points), exc_info=True)
        raise


def search(query_vector: list[float], top_k: int = 5, filters: dict | None = None) -> list[QueryResponse]:
    logger.debug("search: top_k=%d filters=%s", top_k, filters)
    try:
        return get_qdrant().query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=filters,
            with_payload=True,
        ).points
    except Exception:
        logger.error("search failed: top_k=%d", top_k, exc_info=True)
        raise
