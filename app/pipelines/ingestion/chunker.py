from dataclasses import dataclass
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.config import settings
from app.pipelines.ingestion.loader import RawDocument
from app.helpers.logger import get_logger

logger = get_logger(__name__)

MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass
class Chunk:
    chunk_id: str
    doc_filename: str
    doc_type: str
    section: str
    chunk_index: int
    text: str


def _make_chunk(doc: RawDocument, idx: int, section: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.filename}_{idx}",
        doc_filename=doc.filename,
        doc_type=doc.doc_type,
        section=section,
        chunk_index=idx,
        text=text.strip(),
    )


def _chunk_markdown(doc: RawDocument) -> list[Chunk]:
    logger.debug("chunking (markdown): %s (%d chars)", doc.filename, len(doc.content))
    # strip_headers=True removes ### syntax from chunk text; section title is kept in metadata
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MD_HEADERS, strip_headers=True)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    sections = md_splitter.split_text(doc.content)
    chunks = []
    idx = 0
    pending = ""
    pending_title = ""

    for section in sections:
        section_title = (
            section.metadata.get("h3")
            or section.metadata.get("h2")
            or section.metadata.get("h1")
            or "General"
        )
        body = section.page_content.strip()

        # Merge short sections into the next one to avoid near-empty chunks
        if len(body) < settings.min_chunk_length:
            logger.debug("short section merged into next: %s / '%s' (%d chars)", doc.filename, section_title, len(body))
            pending += ("\n\n" + body if pending else body)
            pending_title = pending_title or section_title
            continue

        if pending:
            body = pending + "\n\n" + body
            section_title = pending_title
            pending = ""
            pending_title = ""

        sub_chunks = recursive_splitter.split_text(body)
        for text in sub_chunks:
            if len(text.strip()) < settings.min_chunk_length:
                logger.debug("sub-chunk discarded (too short): %s chunk_%d", doc.filename, idx)
                continue
            chunks.append(_make_chunk(doc, idx, section_title, text))
            idx += 1

    # Flush any remaining pending content
    if pending.strip() and len(pending.strip()) >= settings.min_chunk_length:
        chunks.append(_make_chunk(doc, idx, pending_title or "General", pending))

    logger.debug("chunking done: %s → %d chunks", doc.filename, len(chunks))
    return chunks


def _chunk_generic(doc: RawDocument) -> list[Chunk]:
    logger.debug("chunking (generic): %s (%d chars)", doc.filename, len(doc.content))
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    texts = splitter.split_text(doc.content)

    chunks = [
        _make_chunk(doc, i, "", text)
        for i, text in enumerate(texts)
        if text.strip()
    ]
    logger.debug("chunking done: %s → %d chunks", doc.filename, len(chunks))
    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    if doc.doc_type == "md":
        return _chunk_markdown(doc)
    return _chunk_generic(doc)
