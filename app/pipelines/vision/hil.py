"""Human-in-the-Loop refund review — temporary backend-terminal implementation.

For borderline refund claims the workflow pauses and asks a human reviewer to decide.
Until a reviewer dashboard exists, the decision is read from the server's terminal.

Async-safe: `input()` is blocking, so it runs in a worker thread via `asyncio.to_thread`
to avoid freezing the FastAPI event loop. Caveat: only ONE review can be answered at a
time (they queue on stdin), but other HTTP requests continue to be served meanwhile.
"""

import asyncio

from app.helpers.logger import get_logger

logger = get_logger(__name__)


async def request_human_decision(summary: dict) -> str:
    """Prompt the reviewer in the terminal. Returns "approved" or "rejected"."""
    _print_review_block(summary)

    while True:
        resp = (await asyncio.to_thread(input, "Approve (a) / Reject (r) > ")).strip().lower()
        if resp in ("a", "approve", "approved"):
            logger.info("HIL decision=approved | refund_id=%s", summary.get("refund_id"))
            return "approved"
        if resp in ("r", "reject", "rejected"):
            logger.info("HIL decision=rejected | refund_id=%s", summary.get("refund_id"))
            return "rejected"
        print("  Please type 'a' to approve or 'r' to reject.")


def _print_review_block(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  HUMAN REVIEW REQUIRED — refund claim")
    print("=" * 60)
    print(f"  Refund ID : {summary.get('refund_id', '—')}")
    print(f"  User      : {summary.get('user_email', '—')}")
    print(f"  Claim     : {summary.get('claim', '—')}")
    print(f"  Photo shows: {summary.get('image_description', '—')}")
    print(f"  Issue type: {summary.get('issue_type', '—')}")
    print(f"  Confidence: {summary.get('confidence', 0.0):.2f}")
    print(f"  Verifiable from photo: {summary.get('claim_verifiable_from_photo', '—')}")
    print(f"  Match     : {summary.get('match_reason', '—')}")
    print(f"  Description: {summary.get('description', '—')}")
    print(f"  Evidence  : {summary.get('evidence', '—')}")
    print("-" * 60)
