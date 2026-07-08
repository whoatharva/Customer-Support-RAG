from openai import AzureOpenAI
from app.config import settings
from app.pipelines.ingestion.chunker import Chunk
from app.logger import get_logger

logger = get_logger(__name__)
BATCH_SIZE = 100

_client: AzureOpenAI | None = None


def get_embedding_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version="2024-02-01",
            timeout=60.0,
        )
    return _client


def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    client = get_embedding_client()
    results = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        logger.info("embedding batch %d/%d (%d chunks)", batch_idx + 1, total_batches, len(batch))
        try:
            response = client.embeddings.create(
                model=settings.azure_openai_embedding_deployment,
                input=[c.text for c in batch],
            )
            for chunk, embedding_data in zip(batch, response.data):
                results.append((chunk, embedding_data.embedding))
            logger.debug("batch %d/%d complete (%d embeddings)", batch_idx + 1, total_batches, len(batch))
        except Exception:
            logger.error("embedding failed: batch %d/%d", batch_idx + 1, total_batches, exc_info=True)
            raise

    return results
