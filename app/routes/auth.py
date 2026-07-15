from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from jose import JWTError, jwt
import bcrypt

from app.config import settings
from app.helpers import database
from app.schemas import LoginRequest, TokenResponse, RegisterRequest, RefreshRequest
from app.helpers.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _encode(subject: str, minutes: int, token_type: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return jwt.encode(
        {"sub": subject, "exp": expire, "type": token_type},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def _make_token(email: str) -> TokenResponse:
    return TokenResponse(
        access_token=_encode(email, settings.jwt_expiry_minutes, "access"),
        refresh_token=_encode(email, settings.jwt_refresh_expiry_minutes, "refresh"),
        expires_in=settings.jwt_expiry_minutes * 60,
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest):
    """Create a new user account in Supabase and return a JWT (auto-login)."""
    if database.get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="email already registered")
    password_hash = _hash(body.password)
    database.create_user(name=body.name, email=body.email, password_hash=password_hash)
    logger.info("new user registered: %s", body.email)
    return _make_token(body.email)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Verify credentials against Supabase users table and return a JWT."""
    user = database.get_user_by_email(body.email)
    if not user or not _verify(body.password, user["password_hash"]):
        logger.warning("failed login attempt: username=%s", body.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    logger.info("login success: %s", body.email)
    return _make_token(body.email)


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest):
    """Exchange a valid refresh token for a fresh access + refresh token pair."""
    try:
        payload = jwt.decode(
            body.refresh_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as e:
        logger.warning("refresh token decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    user = database.get_user_by_email(subject)
    if not user:
        logger.warning("refresh attempted with non-existent user: %s", subject)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return _make_token(subject)
