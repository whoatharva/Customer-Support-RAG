"""
Chainlit data layer backed by Supabase.

Enables the left-panel thread history in the Chainlit UI:
  - list_threads  → populates the sidebar with past conversations
  - get_thread    → loads a full conversation when a user clicks a past thread
  - update_thread → called by Chainlit when a session starts or is renamed
  - create_step   → called automatically whenever cl.Message.send() is called

Tables used (see supabase_chainlit_schema.sql):
  cl_threads  — one row per conversation (id = Chainlit thread_id)
  cl_steps    — one row per message (user or assistant)

All methods are async; sync Supabase calls are wrapped with asyncio.to_thread().
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional

from chainlit.data import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

from app.helpers.database import get_db
from app.helpers.logger import get_logger

logger = get_logger(__name__)

_STEP_TYPES = ("user_message", "assistant_message")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(fn):
    """Wrap a zero-argument lambda so it runs in a thread pool."""
    return asyncio.to_thread(fn)


class SupabaseDataLayer(BaseDataLayer):
    """Chainlit BaseDataLayer implementation backed by our Supabase project."""

    def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    # ── Users ─────────────────────────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        # Return None so Chainlit always falls through to create_user(), which
        # receives the original cl.User object (with JWT token in metadata).
        # If we returned a PersistedUser here Chainlit would replace the original
        # user in the session and the token would be lost → KeyError: 'token'.
        return None

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        # Look up the DB record to get the stable UUID, but copy metadata from
        # the original user so the session retains the token and role.
        try:
            result = await _run(
                lambda: get_db().table("users").select("id, email, created_at")
                    .eq("email", user.identifier).limit(1).execute()
            )
            if result.data:
                row = result.data[0]
                return PersistedUser(
                    id=str(row["id"]),
                    identifier=row["email"],
                    createdAt=str(row.get("created_at", _now())),
                    metadata=user.metadata,  # preserve token + role from auth_callback
                )
        except Exception:
            logger.debug("create_user lookup failed for %s", user.identifier, exc_info=True)
        # Fallback: return a minimal PersistedUser so the session is not blocked.
        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=_now(),
            metadata=user.metadata,
        )

    # ── Feedback (thumbs up/down) — not used, stubbed ─────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return feedback.id or ""

    async def delete_feedback(self, feedback_id: str) -> bool:
        return True

    # ── Elements (file attachments) ───────────────────────────────────────────

    async def create_element(self, element) -> None:
        try:
            await _run(lambda: get_db().table("cl_elements").upsert({
                "id": element.id,
                "thread_id": element.thread_id,
                "type": getattr(element, "type", "file"),
                "name": element.name or "",
                "mime": element.mime or "",
                "url": element.url or "",
                "object_key": element.object_key or "",
                "display": element.display or "inline",
                "size": element.size,
                "language": element.language or "",
                "for_id": element.for_id or "",
                "created_at": _now(),
            }, on_conflict="id").execute())
        except Exception:
            logger.warning("create_element failed: %s", element.id, exc_info=True)

    async def get_element(self, thread_id: str, element_id: str):
        try:
            result = await _run(
                lambda: get_db().table("cl_elements").select("*")
                    .eq("id", element_id).eq("thread_id", thread_id).limit(1).execute()
            )
            if result.data:
                r = result.data[0]
                return {
                    "id": r["id"],
                    "threadId": r["thread_id"],
                    "type": r.get("type", "file"),
                    "name": r.get("name", ""),
                    "mime": r.get("mime", ""),
                    "url": r.get("url", ""),
                    "objectKey": r.get("object_key", ""),
                    "display": r.get("display", "inline"),
                    "size": r.get("size"),
                    "language": r.get("language", ""),
                    "forId": r.get("for_id", ""),
                }
        except Exception:
            logger.debug("get_element failed: %s / %s", thread_id, element_id, exc_info=True)
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        try:
            q = get_db().table("cl_elements").delete().eq("id", element_id)
            if thread_id:
                q = q.eq("thread_id", thread_id)
            await _run(lambda: q.execute())
        except Exception:
            logger.debug("delete_element failed: %s", element_id, exc_info=True)

    # ── Steps (individual messages) ───────────────────────────────────────────

    async def create_step(self, step_dict: dict) -> None:
        if step_dict.get("type") not in _STEP_TYPES:
            return  # skip tool-call steps, only persist user/assistant messages
        try:
            await _run(lambda: get_db().table("cl_steps").upsert({
                "id": step_dict["id"],
                "thread_id": step_dict["threadId"],
                "type": step_dict.get("type", ""),
                "name": step_dict.get("name", ""),
                "input": step_dict.get("input") or "",
                "output": step_dict.get("output") or "",
                "metadata": step_dict.get("metadata") or {},
                "is_error": step_dict.get("isError", False),
                "start_time": step_dict.get("start"),
                "end_time": step_dict.get("end"),
                "created_at": step_dict.get("createdAt") or _now(),
            }, on_conflict="id").execute())
        except Exception:
            logger.warning("create_step failed: %s", step_dict.get("id"), exc_info=True)

    async def update_step(self, step_dict: dict) -> None:
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str) -> None:
        try:
            await _run(lambda: get_db().table("cl_steps").delete().eq("id", step_id).execute())
        except Exception:
            logger.debug("delete_step failed: %s", step_id, exc_info=True)

    async def get_favorite_steps(self, user_id: str) -> List[dict]:
        return []

    async def set_step_favorite(self, step_dict: dict, favorite: bool) -> dict:
        return step_dict

    # ── Threads ───────────────────────────────────────────────────────────────

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        # Chainlit's list_threads filters by current_user.id (the UUID from the
        # users table). We must store that UUID in user_id, and the email in
        # user_identifier (used by Chainlit's auth check: thread["userIdentifier"]
        # == user.identifier).
        #
        # user_id argument arrives as the user's EMAIL from our code.
        # When Chainlit's socket calls update_thread on disconnect it passes no
        # user_id — we fall back to cl.context.session.user.
        user_email: Optional[str] = None
        user_uuid: Optional[str] = None

        if user_id:
            user_email = user_id  # we pass email as user_id
        else:
            # Called by Chainlit internally (e.g. on disconnect) — read from context
            try:
                import chainlit as cl_inner
                session_user = cl_inner.context.session.user
                if session_user:
                    user_email = session_user.identifier
                    user_uuid = getattr(session_user, "id", None)
            except Exception:
                pass

        # Resolve email → DB UUID so list_threads filtering by UUID works
        if user_email and not user_uuid:
            try:
                res = await _run(
                    lambda: get_db().table("users").select("id")
                        .eq("email", user_email).limit(1).execute()
                )
                if res.data:
                    user_uuid = str(res.data[0]["id"])
            except Exception:
                user_uuid = user_email  # fallback: store email as id

        # Merge with the existing row: Chainlit calls update_thread multiple times
        # per session (start, rename, disconnect) each passing only a subset of
        # fields. A blind full-row upsert would reset the name to "New Conversation"
        # and wipe session_id metadata. So only overwrite fields explicitly passed.
        existing: dict = {}
        try:
            res = await _run(
                lambda: get_db().table("cl_threads").select("*")
                    .eq("id", thread_id).limit(1).execute()
            )
            if res.data:
                existing = res.data[0]
        except Exception:
            logger.debug("update_thread existing-row read failed: %s", thread_id, exc_info=True)

        # Treat an explicit "New Conversation" as "no name" — Chainlit's internal
        # calls pass that default and would otherwise clobber a real title.
        incoming_name = name if name and name != "New Conversation" else None

        row = {
            "id": thread_id,
            "name": incoming_name or existing.get("name") or "New Conversation",
            "user_identifier": user_email or existing.get("user_identifier"),
            "user_id": user_uuid or existing.get("user_id"),
            "metadata": {**(existing.get("metadata") or {}), **(metadata or {})},
            "tags": tags if tags is not None else (existing.get("tags") or []),
            "updated_at": _now(),
        }

        try:
            await _run(lambda: get_db().table("cl_threads").upsert(row, on_conflict="id").execute())
        except Exception:
            logger.warning("update_thread failed: %s", thread_id, exc_info=True)

    async def get_thread_author(self, thread_id: str) -> str:
        try:
            result = await _run(
                lambda: get_db().table("cl_threads").select("user_identifier")
                    .eq("id", thread_id).limit(1).execute()
            )
            if result.data:
                return result.data[0].get("user_identifier") or ""
        except Exception:
            logger.debug("get_thread_author failed: %s", thread_id, exc_info=True)
        return ""

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        try:
            t_res = await _run(
                lambda: get_db().table("cl_threads").select("*")
                    .eq("id", thread_id).limit(1).execute()
            )
            if not t_res.data:
                return None
            thread = t_res.data[0]

            s_res = await _run(
                lambda: get_db().table("cl_steps").select("*")
                    .eq("thread_id", thread_id)
                    .order("created_at", desc=False).execute()
            )
            steps = [_row_to_step(r) for r in (s_res.data or [])]

            return ThreadDict(
                id=thread["id"],
                createdAt=thread.get("created_at", ""),
                name=thread.get("name", "Conversation"),
                userId=thread.get("user_id"),
                userIdentifier=thread.get("user_identifier"),
                tags=thread.get("tags") or [],
                metadata=thread.get("metadata") or {},
                steps=steps,
                elements=[],
            )
        except Exception:
            logger.debug("get_thread failed: %s", thread_id, exc_info=True)
            return None

    async def delete_thread(self, thread_id: str) -> None:
        try:
            await _run(
                lambda: get_db().table("cl_threads").delete().eq("id", thread_id).execute()
            )
        except Exception:
            logger.debug("delete_thread failed: %s", thread_id, exc_info=True)

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        try:
            def _query():
                q = (
                    get_db().table("cl_threads")
                    .select("id, name, user_identifier, user_id, created_at, updated_at, tags, metadata")
                    .order("updated_at", desc=True)
                    .limit(pagination.first)
                )
                if filters.userId:
                    q = q.eq("user_id", filters.userId)  # UUID, matches current_user.id
                if filters.search:
                    q = q.ilike("name", f"%{filters.search}%")
                if pagination.cursor:
                    q = q.lt("updated_at", pagination.cursor)
                return q.execute()

            result = await _run(_query)
            rows = result.data or []
        except Exception:
            logger.debug("list_threads failed", exc_info=True)
            rows = []

        threads = [
            ThreadDict(
                id=r["id"],
                createdAt=r.get("created_at", ""),
                name=r.get("name") or "Conversation",
                userId=r.get("user_id"),              # UUID
                userIdentifier=r.get("user_identifier"),  # email
                tags=r.get("tags") or [],
                metadata=r.get("metadata") or {},
                steps=[],
                elements=[],
            )
            for r in rows
        ]
        end_cursor = rows[-1]["updated_at"] if rows else None
        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(
                hasNextPage=len(rows) == pagination.first,
                startCursor=rows[0]["updated_at"] if rows else None,
                endCursor=end_cursor,
            ),
        )


def _row_to_step(r: dict) -> dict:
    return {
        "id": r["id"],
        "threadId": r["thread_id"],
        "type": r.get("type", "undefined"),
        "name": r.get("name", ""),
        "input": r.get("input", ""),
        "output": r.get("output", ""),
        "metadata": r.get("metadata") or {},
        "tags": r.get("tags") or [],
        "isError": r.get("is_error", False),
        "createdAt": r.get("created_at", ""),
        "start": r.get("start_time"),
        "end": r.get("end_time"),
    }
