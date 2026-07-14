from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from app.routes import auth, admin, chat
from app.config import settings
from app.helpers.logger import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    logger.info("app started (log_level=%s)", settings.log_level)
    yield


app = FastAPI(title="Customer Support RAG", version="0.1.0", lifespan=lifespan)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {"status": "healthy"}
