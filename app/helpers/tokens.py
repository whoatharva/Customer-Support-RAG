"""JWT encode/decode helpers shared by the API routes and the Chainlit UI.

Kept free of FastAPI imports so the UI layer can mint admin tokens without
depending on route internals.
"""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)


class TokenError(Exception):
    """Raised when a JWT is invalid, expired, or of the wrong type."""


def encode_token(subject: str, minutes: int, token_type: str, role: str = "user") -> str:
    """Mint a signed JWT for `subject` expiring in `minutes`.

    `role` ("user" | "admin") is carried as an explicit claim so authorization
    never depends on the subject string matching a configured admin name.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return jwt.encode(
        {"sub": subject, "exp": expire, "type": token_type, "role": role},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_claims(token: str, expected_type: str) -> dict:
    """Decode and validate a JWT; return its full payload.

    expected_type:
      "access"  — any non-refresh token is accepted
      "refresh" — the token must explicitly be a refresh token

    Raises TokenError with a user-safe message on any failure.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as e:
        logger.warning("JWT decode failed: %s", e)
        raise TokenError("Invalid or expired token")

    token_type = payload.get("type")
    if expected_type == "refresh" and token_type != "refresh":
        raise TokenError("Not a refresh token")
    if expected_type == "access" and token_type == "refresh":
        raise TokenError("Refresh token not accepted here")

    subject = payload.get("sub")
    if not subject:
        raise TokenError("Invalid token subject")
    return payload


def decode_token(token: str, expected_type: str) -> str:
    """Decode and validate a JWT; return its subject (email)."""
    return decode_claims(token, expected_type)["sub"]
