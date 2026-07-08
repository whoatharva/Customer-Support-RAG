"""Pipeline 1 — ingestion tests."""


def test_load_documents():
    # TODO: assert load_documents returns RawDocument list for a temp dir of fixtures
    pass


def test_chunk_document():
    # TODO: assert chunk_document respects CHUNK_SIZE and MIN_CHUNK_LENGTH
    pass


def test_run_ingestion_skips_unchanged():
    # TODO: mock is_document_changed → False, assert files_skipped == 1
    pass
