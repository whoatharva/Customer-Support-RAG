from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from jose import jwt
import bcrypt

from app.config import settings
from app.helpers import database
from app.schemas import LoginRequest, TokenResponse, RegisterRequest
from app.helpers.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _make_token(email: str) -> TokenResponse:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    token = jwt.encode(
        {"sub": email, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    return TokenResponse(access_token=token, expires_in=settings.jwt_expiry_minutes * 60)


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
    user = database.get_user_by_email(body.username)
    if not user or not _verify(body.password, user["password_hash"]):
        logger.warning("failed login attempt: username=%s", body.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    logger.info("login success: %s", body.username)
    return _make_token(body.username)
