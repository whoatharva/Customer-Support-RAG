from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.helpers.tokens import TokenError, decode_token
from app.helpers.logger import get_logger

logger = get_logger(__name__)
bearer_scheme = HTTPBearer()


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """Decode and validate a JWT. Returns the subject (email) on success."""
    try:
        return decode_token(credentials.credentials, expected_type="access")
    except TokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def verify_admin(subject: str = Depends(verify_jwt)) -> str:
    """Restrict an endpoint to the configured admin account only."""
    if subject != settings.admin_username:
        logger.warning("admin check failed: subject=%s", subject)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return subject
