from fastapi import APIRouter, Depends, HTTPException, status

from app.routes.deps import verify_jwt
from app.schemas import UpdateProfileRequest, ProfileResponse
from app.helpers import database
from app.services import profile_service
from app.helpers.logger import get_logger

router = APIRouter(prefix="/profile", tags=["profile"])
logger = get_logger(__name__)


@router.put("/contact", response_model=ProfileResponse)
def update_contact(body: UpdateProfileRequest, user_email: str = Depends(verify_jwt)):
    """Apply a validated phone or address change, then return the fresh profile."""
    logger.info("PUT /profile/contact | user=%s field=%s", user_email, body.field)
    ok, error = profile_service.validate(body.field, body.values)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    try:
        profile_service.apply(user_email, body.field, body.values)
    except Exception as e:
        logger.error("profile update failed | user=%s: %s", user_email, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update profile")
    customer = database.get_customer_full(user_email)
    if not customer:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    addresses = customer.get("addresses") or []
    return ProfileResponse(
        phone=customer.get("phone"),
        address=addresses[0] if addresses else None,
        orders=[],
    )
