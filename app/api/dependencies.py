from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)
bearer_scheme = HTTPBearer()


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        if username != settings.admin_username:
            logger.warning("JWT subject mismatch: expected=%s got=%s", settings.admin_username, username)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
        return username
    except JWTError as e:
        logger.warning("JWT decode failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
