from langfuse import Langfuse
from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)
_client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_base_url,
        )
        logger.info("Langfuse client initialized (host=%s)", settings.langfuse_base_url)
    return _client
