"""Pipeline 4 — refund verification decision-policy tests.

The orchestrator (`analyzer.process_refund_claim`) is exercised across all decision
branches. The vision call (`verifier.verify_image`), the Human-in-the-Loop terminal
prompt (`hil.request_human_decision`), and DB persistence (`database.create_refund_request`)
are all patched so tests are hermetic and never touch network/stdin.
"""

import asyncio
from unittest.mock import MagicMock

from app.pipelines.vision import analyzer
from app.pipelines.vision.verifier import VerificationResult
from app.config import settings


def _result(**overrides) -> VerificationResult:
    base = dict(
        authentic=True,
        authenticity_reason="looks real",
        issue_type="damaged_product",
        issue_matches_claim=True,
        match_reason="crack visible",
        confidence=0.9,
        description="The screen is cracked.",
        evidence="visible crack across display",
        image_description="a laptop with a cracked screen",
        contradicts_claim=False,
        claim_verifiable_from_photo=True,
    )
    base.update(overrides)
    return VerificationResult(**base)


def _run(monkeypatch, result: VerificationResult, hil_decision: str = "approved") -> dict:
    monkeypatch.setattr(analyzer.verifier, "verify_image", lambda *a, **k: result)
    monkeypatch.setattr(analyzer.database, "create_refund_request", lambda **k: {"id": "r1"})
    monkeypatch.setattr(analyzer.database, "ensure_chat_session", lambda *a, **k: None)
    monkeypatch.setattr(analyzer.database, "save_chat_message", lambda *a, **k: None)

    async def _fake_hil(summary):
        return hil_decision
    monkeypatch.setattr(analyzer.hil, "request_human_decision", _fake_hil)

    return asyncio.run(
        analyzer.process_refund_claim(b"imgbytes", "image/jpeg", "screen cracked", "u@x.com", "sess1")
    )


def test_high_confidence_auto_approves(monkeypatch):
    out = _run(monkeypatch, _result(confidence=0.95))
    assert out["decision"] == "approved"
    assert out["issue_type"] == "damaged_product"


def test_inauthentic_image_rejected(monkeypatch):
    out = _run(monkeypatch, _result(authentic=False, authenticity_reason="looks AI-generated"))
    assert out["decision"] == "rejected"
    assert "AI-generated" in out["reason"]


def test_issue_mismatch_rejected(monkeypatch):
    out = _run(monkeypatch, _result(issue_matches_claim=False, match_reason="no damage visible"))
    assert out["decision"] == "rejected"
    assert "no damage visible" in out["reason"]


def test_low_confidence_rejected(monkeypatch):
    below = settings.refund_auto_reject_threshold - 0.05
    out = _run(monkeypatch, _result(confidence=below))
    assert out["decision"] == "rejected"


def test_borderline_goes_to_hil_approve(monkeypatch):
    mid = (settings.refund_auto_approve_threshold + settings.refund_auto_reject_threshold) / 2
    out = _run(monkeypatch, _result(confidence=mid), hil_decision="approved")
    assert out["decision"] == "approved"
    assert "reviewer" in out["reason"].lower()


def test_borderline_goes_to_hil_reject(monkeypatch):
    mid = (settings.refund_auto_approve_threshold + settings.refund_auto_reject_threshold) / 2
    out = _run(monkeypatch, _result(confidence=mid), hil_decision="rejected")
    assert out["decision"] == "rejected"
    assert "reviewer" in out["reason"].lower()


def test_contradiction_auto_rejected(monkeypatch):
    # The laptop-in-box case: photo shows the item present, claim says it's missing.
    hil_spy = MagicMock()
    monkeypatch.setattr(analyzer.verifier, "verify_image",
                        lambda *a, **k: _result(contradicts_claim=True, confidence=0.95,
                                                match_reason="laptop is clearly present in the box"))
    monkeypatch.setattr(analyzer.database, "create_refund_request", lambda **k: {"id": "r1"})
    monkeypatch.setattr(analyzer.database, "ensure_chat_session", lambda *a, **k: None)
    monkeypatch.setattr(analyzer.database, "save_chat_message", lambda *a, **k: None)
    monkeypatch.setattr(analyzer.hil, "request_human_decision", hil_spy)

    out = asyncio.run(
        analyzer.process_refund_claim(b"img", "image/jpeg", "laptop missing", "u@x.com", "s1")
    )
    assert out["decision"] == "rejected"
    assert hil_spy.call_count == 0  # auto-rejected, never reaches HIL


def test_unverifiable_absence_routes_to_hil(monkeypatch):
    # High confidence but the claim can't be proven from a photo → must NOT auto-approve.
    out = _run(
        monkeypatch,
        _result(confidence=0.95, claim_verifiable_from_photo=False),
        hil_decision="rejected",
    )
    assert out["decision"] == "rejected"  # reviewer decided, not auto-approved


def test_parse_failure_not_auto_approved(monkeypatch):
    # A malformed vision response (safe-fallback) must never auto-approve.
    from app.pipelines.vision import verifier
    fallback = verifier._parse_response("not json at all")
    out = _run(monkeypatch, fallback, hil_decision="rejected")
    assert out["decision"] == "rejected"


def test_decision_is_persisted(monkeypatch):
    spy = MagicMock(return_value={"id": "r1"})
    save_spy = MagicMock()
    monkeypatch.setattr(analyzer.verifier, "verify_image", lambda *a, **k: _result())
    monkeypatch.setattr(analyzer.database, "create_refund_request", spy)
    monkeypatch.setattr(analyzer.database, "ensure_chat_session", lambda *a, **k: None)
    monkeypatch.setattr(analyzer.database, "save_chat_message", save_spy)
    asyncio.run(
        analyzer.process_refund_claim(b"img", "image/jpeg", "cracked", "u@x.com", "s1")
    )
    assert spy.call_count == 1
    assert spy.call_args.kwargs["decision"] == "approved"
    assert spy.call_args.kwargs["decided_by"] == "auto"
    # Refund turn saved to chat history (user + assistant messages).
    assert save_spy.call_count == 2
