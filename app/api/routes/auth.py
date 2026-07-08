from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from jose import jwt
from app.config import settings
from app.schemas import LoginRequest, TokenResponse
from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    if body.username != settings.admin_username or body.password != settings.admin_password:
        logger.warning("failed login attempt: username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    token = jwt.encode(
        {"sub": settings.admin_username, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    logger.info("successful login: username=%s", body.username)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expiry_minutes * 60)
