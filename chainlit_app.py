import os
import shutil
import tempfile
import httpx
import chainlit as cl

API_BASE = "http://localhost:8000/api/v1"
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}


# ── Auth ─────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{API_BASE}/auth/login",
                json={"username": username, "password": password},
            )
            if res.status_code == 200:
                token = res.json()["access_token"]
                return cl.User(identifier=username, metadata={"token": token})
        except Exception:
            pass
    return None


def get_token() -> str:
    user = cl.user_session.get("user")
    return user.metadata["token"]


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


# ── Chat start ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("browse_path", os.path.expanduser("~"))
    await cl.Message(
        content=(
            "**Customer Support RAG — Admin UI**\n\n"
            "**Options:**\n"
            "- Drag & drop `.md`, `.pdf`, or `.docx` files to ingest them\n"
            "- `/browse` — open folder browser to pick a folder\n"
            "- `/ingest <path>` — ingest a folder by path directly\n"
            "- `/status` — view last ingestion runs\n"
        )
    ).send()


# ── File upload (drag & drop) ─────────────────────────────────────────────────

async def handle_file_upload(message: cl.Message):
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


# ── Folder browser ────────────────────────────────────────────────────────────

async def show_folder_browser(path: str):
    cl.user_session.set("browse_path", path)

    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        await cl.Message(content=f"Permission denied: `{path}`").send()
        return

    actions = []

    # Parent directory
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

    # Ingest current folder action
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
    # File upload via drag & drop
    if message.elements:
        await handle_file_upload(message)
        return

    text = message.content.strip()

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
        await cl.Message(
            content="Pipeline 2 & 3 not implemented yet.\n\nTry:\n- `/browse` — folder browser\n- `/ingest <path>` — ingest by path\n- `/status` — ingestion history\n- Or drag & drop files here"
        ).send()
