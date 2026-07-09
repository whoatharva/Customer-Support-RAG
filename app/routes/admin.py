from fastapi import APIRouter, Depends, HTTPException, status
from app.routes.deps import verify_admin as verify_jwt
from app.schemas import IngestRequest, IngestResponse, IngestionStatusResponse, IngestError
from app.helpers import database
from app.pipelines.ingestion.indexer import run_ingestion
from app.helpers.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(body: IngestRequest, _: str = Depends(verify_jwt)):
    logger.info("ingest request received: path=%s", body.path)
    try:
        result = run_ingestion(body.path)
    except FileNotFoundError as e:
        logger.error("ingest path not found: %s", body.path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    logger.info(
        "ingest completed: run_id=%s processed=%d skipped=%d chunks=%d errors=%d",
        result["run_id"], result["files_processed"], result["files_skipped"],
        result["chunks_created"], len(result["errors"]),
    )
    return IngestResponse(
        run_id=result["run_id"],
        files_processed=result["files_processed"],
        files_skipped=result["files_skipped"],
        chunks_created=result["chunks_created"],
        errors=[IngestError(**e) for e in result["errors"]],
    )


@router.get("/ingestion/status", response_model=list[IngestionStatusResponse])
def ingestion_status(_: str = Depends(verify_jwt)):
    return database.get_ingestion_logs()
