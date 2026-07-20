from app.config import settings
from app.llm import client as llm_client
from app.pipelines.ingestion.chunker import Chunk
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    results = []
    batch_size = settings.embedding_batch_size
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        logger.info("embedding batch %d/%d (%d chunks)", batch_idx + 1, total_batches, len(batch))
        try:
            vectors = llm_client.embed([c.text for c in batch])
            results.extend(zip(batch, vectors))
            logger.debug("batch %d/%d complete (%d embeddings)", batch_idx + 1, total_batches, len(batch))
        except Exception:
            logger.error("embedding failed: batch %d/%d", batch_idx + 1, total_batches, exc_info=True)
            raise

    return results
