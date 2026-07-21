from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.helpers.tokens import TokenError, decode_claims
from app.helpers.logger import get_logger

logger = get_logger(__name__)
bearer_scheme = HTTPBearer()


def verify_claims(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Decode and validate a JWT. Returns the full claims payload on success."""
    try:
        return decode_claims(credentials.credentials, expected_type="access")
    except TokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


def verify_jwt(claims: dict = Depends(verify_claims)) -> str:
    """Decode and validate a JWT. Returns the subject (email) on success."""
    return claims["sub"]


def verify_admin(claims: dict = Depends(verify_claims)) -> str:
    """Restrict an endpoint to tokens carrying an explicit admin role claim."""
    if claims.get("role") != "admin":
        logger.warning("admin check failed: subject=%s role=%s", claims.get("sub"), claims.get("role"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return claims["sub"]
