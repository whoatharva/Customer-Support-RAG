from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)
bearer_scheme = HTTPBearer()


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """Decode and validate a JWT. Returns the subject (email) on success."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if payload.get("type") == "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token not accepted here")
        subject = payload.get("sub")
        if not subject:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
        return subject
    except JWTError as e:
        logger.warning("JWT decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def verify_admin(subject: str = Depends(verify_jwt)) -> str:
    """Restrict an endpoint to the configured admin account only."""
    if subject != settings.admin_username:
        logger.warning("admin check failed: subject=%s", subject)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return subject
