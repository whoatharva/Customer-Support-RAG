# Helper-Extraction Refactor — Audit & Plan

## Context

Follow-up to the dead-code cleanup (completed earlier today, 47 tests green). This pass identifies code that should be **extracted into helper modules**: repeated logic, oversized functions, generic utilities living in the wrong layer, and SRP violations. Two thorough exploration passes covered (1) UI/routes/data-layer and (2) services/pipelines/helpers. Every finding was verified with exact occurrence counts, and **every extraction was checked against the test suite's monkeypatch points** (tests patch module attributes like `retriever.vector_store`, `analyzer.database`, `gates.lightweight` — extractions must not move those names).

**Scope decision (user-confirmed): helpers only.** The optional `chainlit_ui/` package split of chainlit_app.py (560 lines, 5 mixed concerns) is documented at the bottom as future work but NOT part of this plan.

## Findings Table

| File | Function/Code Block | Suggested Helper | Reason | Expected Benefit |
|---|---|---|---|---|
| chat_service.py:318-326, profile_service.py:40-45, processor.py:53-59, verifier.py:72-76 | strip fences → `json.loads` → catch → safe default (×4) | `parse_json_lenient()` in existing `app/llm/parsing.py` | Same lenient-JSON recovery re-implemented 4× | One hardened parse path; consistent logging |
| chainlit_app.py — 6 try/except blocks + 7 progress-msg pairs | `httpx.HTTPStatusError`→"API error {code}", `Exception`→"Error: {e}", `msg.update()` | `progress()` + `api_errors()` context manager in a new `chainlit_helpers.py` | Verbatim error rendering ×6; `cl.Message`+send ×7 | ~30 lines removed; single place for error UX |
| chainlit_app.py:164-184 | Confirm/Cancel action rendering (profile_update vs order_cancel — structurally identical) | `render_confirmation()` in `chainlit_helpers.py` | Two identical action-button blocks driven by `data["action"]` | Adding a 3rd confirmable action becomes one dict entry |
| chainlit_app.py:447-498 | `on_confirm_profile` vs `on_confirm_order_cancel` — same skeleton (remove → progress → request → special-status → success/fail) | `confirm_action_flow()` in `chainlit_helpers.py` | Copy-paste callback shape ×2 (+2 trivial dismiss callbacks) | ~25 lines removed |
| routes/auth.py:66-77, routes/deps.py:14-24 | `jwt.decode` + JWTError→401 + type check + sub check (×2) | `decode_token()` next to `encode_token` in new `app/helpers/tokens.py`; move `encode_token` there too | Duplicate decode/validation; also fixes layering smell (chainlit_app imports `encode_token` from a route module) | One token module; UI no longer imports route internals |
| chainlit_data_layer.py — ~13 try/except blocks, 5 fetch-one blocks | `try: await _run(...) except: logger...` and `.select().eq().limit(1)` → first row | module-private `_safe()` and `_fetch_one()` in same file | Same safe-query wrapper ×13, fetch-one ×5 | File drops ~80-100 lines; uniform failure logging |
| retriever.py:54-180 `generate_response` (127 lines) | ⑤ context assembly, ⑥ generation+usage bookkeeping, ⑧ invoice citations, ⑧ confidence formula | private `_assemble_context`, `_generate_answer`, `_build_invoice_citations`, `_compute_confidence` (same module) | 8-step god function; `_build_faq_citations` already extracted, invoice twin left inline | 127→~60 lines; all `retriever.*` patch points intact |
| chat_service.py:56-94 `handle_chat` profile branch | 38-line nested block with 3 exits | private `_handle_profile()` (returns `ChatResponse \| None`; None → fall through to RAG) | `_handle_cancel`/`_handle_multi` already extracted; profile branch inconsistently inline | handle_chat 67→~40 lines |
| chat_service.py:333-338 `_confirm_prompt` | Profile-update confirmation wording | Move to `profile_service.confirm_prompt()` | Pure profile UX; input is `profile_service.detect()` output; duplicates `apply()`'s wording idiom | Fixes SRP leak; wording lives beside `apply()` |
| chat_service.py:28-35 `_persist_turn`, analyzer.py:137-142 `_finalize` persistence | ensure_session + save user + save assistant, best-effort (×2) | `persist_chat_turn()` in existing `app/helpers/database.py` | Same 3-call sequence duplicated | Dedup; `analyzer.database.persist_chat_turn` stays patchable |
| database.py ×6, order_lookup.py ×2 | `result.data[0] if result.data else default` (×8) | `first_row()` in `app/helpers/database.py` | Expression-level dedup ×8 | Consistent empty-result handling |
| order_cancellation.py:70, profile_service.py:35, chat_service.py:289+305 | `any(kw in q.lower() for kw in TUPLE)` (×4) | `contains_any()` in new `app/helpers/text.py`; public gate fns stay as one-line wrappers | Same keyword-gate idiom ×4 with inconsistent lowercasing | Standardized casing; gates stay patchable in place |
| date_facts.py:52-96 | 3× "today / 1 day ago / N days ago" pluralization ladders | private `_days_delta_phrase()` (same module) | Same if/elif ladder ×3 | ~25 lines; one place for wording |
| chat_service.py:129, profile_service.py:95 | `", ".join(i.get("name","") for i in order["items"])` (×2) | `format_order_items()` in existing `app/helpers/order_lookup.py` | Same join idiom + `f" ({items})"` suffix ×2 | Minor dedup |
| database.py:139-146 `get_invoices_for_user` | Defined under "Chat history" section, re-exported via order_lookup shim | Move definition into `app/helpers/order_lookup.py` | Business-data function in wrong module + noqa re-export | Removes shim; clarifies domain boundary |
| analyzer.py:68-74, 90-93 | 2 identical HIL-decision blocks; `_finalize` threads 7 positional args ×6 | private `_decide_via_hil()`; optional small ctx dataclass | Duplicate HIL wiring | Readability; `analyzer.hil` patch point safe |
| processor.py:35-36, retriever.py:71-95, chat_service.py:232-239 | Langfuse `trace`/`span.end` boilerplate (3 traces, 4 span pairs) | **Deferred** — see "Explicitly not doing" | Tests patch `retriever.get_langfuse`/`processor.get_langfuse` by name; helper indirection risks silent patch bypass | — |

## Explicitly NOT doing (verified anti-recommendations)

- **Langfuse `traced_span` helper** — tests patch `get_langfuse` per consuming module; a helper would need coordinated conftest changes for marginal gain. Defer.
- **Splitting `database.py` by domain** — would break 12 `setattr(X.database, ...)` patch points unless a re-export facade is kept; file is 250 lines with clear section comments. Keep.
- **Folding `lightweight.call()` into a call-and-parse helper** — breaks `gates.lightweight` patching. Parse-only helper instead (finding 1).
- **`@lazy_singleton` decorator** for the 4 `global _client` blocks — negative risk/benefit.
- **`chainlit_ui/` package split** — user chose helpers-only scope. Future work: split chainlit_app.py into api_client / auth / admin_ui / chat_ui modules.

## Implementation order

1. **New `app/helpers/tokens.py`** — move `encode_token` from routes/auth.py, add `decode_token(token, expected_type)`. Update: auth.py (`_make_token`, `refresh`), deps.py (`verify_jwt`), chainlit_app.py import.
2. **Extend `app/llm/parsing.py`** — add `parse_json_lenient(raw, default=None, *, slice_from=None)`. Update the 4 call sites; keep each caller's default shape and its own `lightweight.call`/`llm.chat` invocation in place.
3. **New `app/helpers/text.py`** — `contains_any(text, terms)`. Rewire the 4 gate sites; keep public gate functions where they are.
4. **`app/helpers/database.py`** — add `first_row(result, default=None)` (use at 8 sites) and `persist_chat_turn(...)` (delegate from `chat_service._persist_turn` and `analyzer._finalize`); move `get_invoices_for_user` to order_lookup.py (drop the noqa shim).
5. **`app/helpers/order_lookup.py`** — add `format_order_items(order)`; use in chat_service + profile_service.
6. **`date_facts.py`** — internal `_days_delta_phrase()` refactor of the 3 ladders.
7. **`retriever.py`** — split `generate_response` into the 4 private helpers. CRITICAL: helpers must keep referencing module-level `llm`, `vector_store`, `gates`, `date_facts`, `order_lookup` names (no import-time rebinding) so test patches keep working.
8. **`chat_service.py`** — extract `_handle_profile()`; move `_confirm_prompt` → `profile_service.confirm_prompt()`.
9. **`analyzer.py`** — `_decide_via_hil()` extraction.
10. **New `chainlit_helpers.py`** (UI-side, next to chainlit_app.py) — `progress()`, `api_errors()` ctx manager, `render_confirmation()`, `confirm_action_flow()`. Rewire the 6+7+2+2 sites in chainlit_app.py.
11. **`chainlit_data_layer.py`** — module-private `_safe()` + `_fetch_one()`, collapse the 13 try/except blocks.

## Verification

- `python -m pytest` after each step — suite must stay at 47 passed (tests patch module attributes; any failure signals a broken patch point, fix before proceeding).
- AST/import sweep: compile all files + import `app.app`, `chainlit_app` with dummy env (same check used in the dead-code pass).
- Manual smoke test at the end: start backend (`python main.py`) + `chainlit run chainlit_app.py`; exercise one chat turn with citations, one profile update confirm, one order-cancel confirm, one admin `/ingest` + `/status`, one image-refund flow (these cover every rewired UI path).
- Behavior must be identical — this is a pure structural refactor; any user-visible string change is a bug (exception: none planned).
