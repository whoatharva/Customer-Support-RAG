import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
import httpx
from jose import jwt
import chainlit as cl
import chainlit.data as cl_data
from dotenv import load_dotenv
from app.llm.prompts import SUPPORT_CONTACT
from app.chainlit_data_layer import SupabaseDataLayer
from app.config import settings

load_dotenv()

cl_data._data_layer = SupabaseDataLayer()

API_BASE = "http://localhost:8000/api/v1"
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}
INVOICE_EXTENSIONS = {".md"}


def _mint_jwt(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode({"sub": subject, "exp": expire}, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# ── Auth ─────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    username = username.strip()

    # Admin login — validated locally against config, then mint a JWT for API calls
    if username == settings.admin_username:
        if password == settings.admin_password:
            token = _mint_jwt(settings.admin_username)
            return cl.User(identifier=username, metadata={"token": token, "role": "admin"})
        return None

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{API_BASE}/auth/login",
                json={"email": username, "password": password},
            )
            if res.status_code == 200:
                token = res.json()["access_token"]
                return cl.User(identifier=username, metadata={"token": token, "role": "user"})
            print(f"[auth] login failed: status={res.status_code} body={res.text}")
        except Exception as e:
            print(f"[auth] exception calling API: {e}")
    return None


def get_token() -> str:
    return cl.user_session.get("user").metadata.get("token", "")


def is_admin() -> bool:
    user = cl.user_session.get("user")
    return user is not None and user.metadata.get("role") == "admin"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def call_ingest(path: str) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        res = await client.post(
            f"{API_BASE}/admin/ingest",
            json={"path": path},
            headers={"Authorization": f"Bearer {get_token()}"},
        )
        res.raise_for_status()
        return res.json()


def format_ingest_result(data: dict) -> str:
    lines = [
        "**Ingestion complete**",
        f"- Files processed: `{data['files_processed']}`",
        f"- Files skipped (unchanged): `{data['files_skipped']}`",
        f"- Chunks created: `{data['chunks_created']}`",
        f"- Run ID: `{data['run_id']}`",
    ]
    if data["errors"]:
        lines.append("\n**Errors:**")
        for e in data["errors"]:
            lines.append(f"- `{e['file']}`: {e['reason']}")
    return "\n".join(lines)


async def handle_chat_query(text: str):
    session_id = cl.user_session.get("session_id")
    invoice_ids = cl.user_session.get("invoice_ids") or []

    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                f"{API_BASE}/chat/query",
                json={"session_id": session_id, "query": text, "invoice_ids": invoice_ids or None},
                headers={"Authorization": f"Bearer {get_token()}"},
            )
            res.raise_for_status()
            data = res.json()

        answer = data["answer"]
        should_escalate = data["should_escalate"]
        citations = data.get("citations", [])

        lines = [answer, ""]
        seen: set[str] = set()
        source_lines: list[str] = []
        for c in citations:
            is_invoice = c["source_document"].startswith("INVOICE:")
            # Always show invoice citations; hide FAQ chunks below relevance threshold
            if not is_invoice and c.get("score", 0) < 0.45:
                continue
            key = f"{c['source_document']} / {c['section']}"
            if key not in seen:
                seen.add(key)
                source_lines.append(f"- `{c['source_document']}` — {c['section']}")
        if source_lines:
            lines.append("**Sources:**")
            lines.extend(source_lines)
        if should_escalate:
            lines.append(f"\n> ⚠️ I'm not fully confident in this answer — please verify with **{SUPPORT_CONTACT}**.")

        msg.content = "\n".join(lines)
        await msg.update()

    except httpx.HTTPStatusError as e:
        msg.content = f"API error {e.response.status_code}: {e.response.text}"
        await msg.update()
    except Exception as e:
        msg.content = f"Error: {e}"
        await msg.update()


# ── Chat start ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("invoice_ids", [])

    if is_admin():
        cl.user_session.set("browse_path", os.path.expanduser("~"))
        await cl.Message(
            content=(
                "**Customer Support RAG — Admin**\n\n"
                "**Chat:** Type any question to query the knowledge base.\n\n"
                "**Ingestion commands:**\n"
                "- Drag & drop `.md`, `.pdf`, or `.docx` files to ingest them\n"
                "- `/browse` — open folder browser to pick a folder\n"
                "- `/ingest <path>` — ingest a folder by path directly\n"
                "- `/status` — view last ingestion runs\n"
            )
        ).send()
    else:
        await cl.Message(
            content=(
                "**Customer Support**\n\n"
                "Hi! How can I help you today?\n\n"
                "You can upload your **invoice** (`.md` file) to get help with a specific order, "
                "or just type your question directly."
            )
        ).send()


# ── File upload (drag & drop) ─────────────────────────────────────────────────

async def handle_file_upload(message: cl.Message):
    if is_admin():
        await _handle_admin_upload(message)
    else:
        await _handle_user_invoice_upload(message)


async def _handle_admin_upload(message: cl.Message):
    files = [e for e in message.elements if hasattr(e, "path") and
             os.path.splitext(e.name)[1].lower() in SUPPORTED_EXTENSIONS]

    if not files:
        await cl.Message(
            content="No supported files found. Upload `.md`, `.pdf`, or `.docx` files."
        ).send()
        return

    msg = cl.Message(content=f"Uploading {len(files)} file(s) and ingesting...")
    await msg.send()

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        for f in files:
            shutil.copy(f.path, os.path.join(tmp_dir, f.name))
        data = await call_ingest(tmp_dir)
        msg.content = format_ingest_result(data)
        await msg.update()
    except httpx.HTTPStatusError as e:
        msg.content = f"API error {e.response.status_code}: {e.response.text}"
        await msg.update()
    except Exception as e:
        msg.content = f"Error: {e}"
        await msg.update()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _handle_user_invoice_upload(message: cl.Message):
    invoices = [e for e in message.elements if hasattr(e, "path") and
                os.path.splitext(e.name)[1].lower() in INVOICE_EXTENSIONS]

    if not invoices:
        await cl.Message(
            content="Please upload your invoice as a `.md` file."
        ).send()
        return

    user_email = cl.user_session.get("user").identifier
    _BASE = os.path.join(os.path.dirname(__file__), "data", "invoices")
    active_ids: list[str] = list(cl.user_session.get("invoice_ids") or [])

    for f in invoices:
        invoice_id = os.path.splitext(f.name)[0]
        invoice_path = os.path.join(_BASE, user_email, f.name)
        if not os.path.exists(invoice_path):
            await cl.Message(
                content=(
                    f"⚠️ Invoice **{invoice_id}** is not associated with your account. "
                    "Please upload one of your own invoices."
                )
            ).send()
            continue
        if invoice_id not in active_ids:
            active_ids.append(invoice_id)

    if not active_ids:
        return

    cl.user_session.set("invoice_ids", active_ids)
    ids_str = ", ".join(f"**{i}**" for i in active_ids)
    await cl.Message(
        content=f"Active invoices: {ids_str}. You can now ask me about your orders."
    ).send()


# ── Folder browser (admin only) ───────────────────────────────────────────────

async def show_folder_browser(path: str):
    cl.user_session.set("browse_path", path)

    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        await cl.Message(content=f"Permission denied: `{path}`").send()
        return

    actions = []

    parent = os.path.dirname(path)
    if parent != path:
        actions.append(cl.Action(name="browse_up", value=parent, label=".. (go up)"))

    for entry in entries:
        if entry.is_dir():
            actions.append(cl.Action(
                name="browse_cd",
                value=entry.path,
                label=f"📁 {entry.name}",
            ))
        elif os.path.splitext(entry.name)[1].lower() in SUPPORTED_EXTENSIONS:
            actions.append(cl.Action(
                name="browse_file",
                value=entry.path,
                label=f"📄 {entry.name}",
            ))

    actions.append(cl.Action(
        name="browse_ingest",
        value=path,
        label=f"✅ Ingest this folder: {os.path.basename(path) or path}",
    ))

    if not actions:
        await cl.Message(content=f"No supported files or folders in `{path}`.").send()
        return

    await cl.Message(
        content=f"**Browsing:** `{path}`\n\nSelect a folder to open or ingest:",
        actions=actions,
    ).send()


@cl.action_callback("browse_cd")
async def on_browse_cd(action: cl.Action):
    await show_folder_browser(action.value)


@cl.action_callback("browse_up")
async def on_browse_up(action: cl.Action):
    await show_folder_browser(action.value)


@cl.action_callback("browse_file")
async def on_browse_file(action: cl.Action):
    await cl.Message(content=f"Selected file: `{action.value}`\nTo ingest it, ingest its parent folder.").send()


@cl.action_callback("browse_ingest")
async def on_browse_ingest(action: cl.Action):
    path = action.value
    msg = cl.Message(content=f"Ingesting folder `{path}`...")
    await msg.send()
    try:
        data = await call_ingest(path)
        msg.content = format_ingest_result(data)
        await msg.update()
    except httpx.HTTPStatusError as e:
        msg.content = f"API error {e.response.status_code}: {e.response.text}"
        await msg.update()
    except Exception as e:
        msg.content = f"Error: {e}"
        await msg.update()


# ── Message handler ───────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    if message.elements:
        await handle_file_upload(message)
        return

    text = message.content.strip()

    if not is_admin():
        await handle_chat_query(text)
        return

    # Admin-only commands
    if text == "/browse":
        current = cl.user_session.get("browse_path") or os.path.expanduser("~")
        await show_folder_browser(current)

    elif text.startswith("/ingest"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            await cl.Message(content="Usage: `/ingest <path>`").send()
            return
        path = parts[1].strip()
        msg = cl.Message(content=f"Ingesting `{path}`...")
        await msg.send()
        try:
            data = await call_ingest(path)
            msg.content = format_ingest_result(data)
            await msg.update()
        except FileNotFoundError:
            msg.content = f"Path not found: `{path}`"
            await msg.update()
        except httpx.HTTPStatusError as e:
            msg.content = f"API error {e.response.status_code}: {e.response.text}"
            await msg.update()
        except Exception as e:
            msg.content = f"Error: {e}"
            await msg.update()

    elif text == "/status":
        msg = cl.Message(content="Fetching ingestion history...")
        await msg.send()
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    f"{API_BASE}/admin/ingestion/status",
                    headers={"Authorization": f"Bearer {get_token()}"},
                )
                res.raise_for_status()
                runs = res.json()

            if not runs:
                msg.content = "No ingestion runs found."
                await msg.update()
                return

            lines = ["**Last ingestion runs:**\n"]
            for run in runs[:10]:
                error_count = len(run.get("errors") or [])
                lines.append(
                    f"- `{run['run_id'][:8]}...` | "
                    f"processed: **{run['files_processed']}** | "
                    f"skipped: {run['files_skipped']} | "
                    f"chunks: **{run['chunks_created']}** | "
                    f"errors: {'⚠️ ' + str(error_count) if error_count else '✅ 0'}"
                )
            msg.content = "\n".join(lines)
            await msg.update()
        except Exception as e:
            msg.content = f"Error: {e}"
            await msg.update()

    else:
        await handle_chat_query(text)
