from fastapi import FastAPI, status
from app.api.routes import auth, admin, chat
from app.config import settings
from app.logger import configure_logging, get_logger

logger = get_logger(__name__)

app = FastAPI(title="Customer Support RAG", version="0.1.0")

app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    configure_logging(settings.log_level)
    logger.info("app started (log_level=%s)", settings.log_level)


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {"status": "healthy"}
