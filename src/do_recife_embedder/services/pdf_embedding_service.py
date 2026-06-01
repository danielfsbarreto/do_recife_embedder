# =============================================================================
# PDF EMBEDDING SERVICE
# -----------------------------------------------------------------------------
# This module is the "ingestion" (indexing) half of a RAG (Retrieval-Augmented
# Generation) system. Its job: take local PDF files, turn their text into
# vector embeddings, and store those vectors in MongoDB Atlas so they can later
# be searched by semantic similarity.
#
# The end-to-end pipeline for each PDF is:
#     extract text (per page) -> split into chunks -> embed each chunk
#     -> store {text, embedding, metadata} documents in MongoDB
#
# Why chunks? Embedding models have a token limit and retrieval works better on
# small, focused passages than on whole documents. Why per page? It keeps memory
# low (we never load the whole PDF) and gives us a natural unit for resuming.
# =============================================================================

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

# Third-party clients/helpers:
from openai import OpenAI  # creates the embedding vectors
from pymongo import ASCENDING, MongoClient  # talks to MongoDB Atlas
from pymongo.collection import Collection  # type hint for a single collection
from pymongo.operations import SearchIndexModel  # defines the Atlas vector index
from tqdm import tqdm  # progress bar in the terminal

# Our own helpers (see the utils package):
#   DoIssueParser/DoIssue -> pull the "edição/issue" info out of a DO PDF
#   PdfExtractor          -> stream a PDF page by page
#   TextChunker           -> split page text into token-sized chunks
from do_recife_embedder.utils import DoIssue, DoIssueParser, PdfExtractor, TextChunker

# --- Embedding configuration -------------------------------------------------
# The model and its output size must match the vector index we create in Atlas.
# "text-embedding-3-large" returns 3072-dimensional vectors.
_EMBEDDING_MODEL = "text-embedding-3-large"
_EMBEDDING_DIMENSIONS = 3072

# --- Atlas vector search index settings --------------------------------------
_VECTOR_INDEX_NAME = "vector_index"
_VECTOR_INDEX_POLL_INTERVAL = 5  # seconds between "is the index ready?" checks
_VECTOR_INDEX_POLL_TIMEOUT = 300  # give up waiting after 5 minutes

# Pre-compiled regex that collapses any run of whitespace into a single space.
# Used to normalize page text before hashing (so trivial spacing differences
# don't change the hash).
_WHITESPACE_RE = re.compile(r"\s+")

# --- Storage targets (hardcoded on purpose for this training project) --------
# We write to TWO collections to compare retrieval quality with and without
# metadata. Both get the SAME text and the SAME vector; only the stored shape
# differs.
_DATABASE_NAME = "do-recife"
_COLLECTION_PLAIN = "do-recife-rag"  # without metadata: {_id, text, embedding}
_COLLECTION_ENRICHED = "do-recife-rag-enriched"  # adds a rich `metadata` object

# --- Concurrency -------------------------------------------------------------
# How many PDFs to embed at the same time. The per-file work is I/O-bound (it
# spends most of its time waiting on the OpenAI and MongoDB network calls), so
# running a few files on separate threads overlaps that waiting and speeds the
# whole run up. A bounded thread pool of this size acts as our semaphore: never
# more than _MAX_PARALLEL_FILES files run concurrently. Tune it to taste and to
# your OpenAI rate limits.
_MAX_PARALLEL_FILES = 5


class PdfEmbeddingService:
    """Embed local PDFs into two MongoDB Atlas Vector Search collections.

    Each page is extracted, chunked, and embedded exactly once; the resulting
    vector is written to both collections so they share identical embeddings and
    text and differ only in stored shape:

    - ``do-recife-rag`` (plain): ``{_id, text, embedding}`` only.
    - ``do-recife-rag-enriched``: same plus a ``metadata`` document, including the
      parsed Diário Oficial issue info (issue number, year, edition date, extra).

    The pipeline streams per file and per page so arbitrarily large PDFs never
    need to be fully buffered. It is resumable and idempotent: the ``_id`` of each
    chunk is deterministic (``f"{page_hash}:{idx}"``), so re-runs detect already
    embedded pages (per collection) and skip them without re-embedding.

    Files are processed concurrently by a bounded thread pool (at most
    ``_MAX_PARALLEL_FILES`` at once). Threads work well here because each file
    spends most of its time waiting on network I/O (OpenAI + MongoDB). The shared
    clients (``MongoClient``, ``OpenAI``) and the ``TextChunker`` are all safe to
    use from multiple threads.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        embed_batch_size: int = 32,
    ):
        # Secrets/connection details come from environment variables (.env), never
        # hardcoded. We fail fast with a clear message if they are missing.
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not connection_string:
            raise ValueError(
                "MongoDB connection string must be provided as environment "
                "variable MONGODB_CONNECTION_STRING"
            )
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key must be provided as environment variable OPENAI_API_KEY"
            )

        # Where to look for *.pdf files, and how many chunks to embed per API call.
        self._data_dir = Path(data_dir)
        self._embed_batch_size = embed_batch_size

        # Open the clients once and reuse them. MongoClient is lazy: it does not
        # actually connect until the first real operation.
        self._mongo_client = MongoClient(connection_string)
        self._db = self._mongo_client[_DATABASE_NAME]
        self._plain = self._db[_COLLECTION_PLAIN]  # handle to the plain collection
        self._enriched = self._db[_COLLECTION_ENRICHED]  # handle to the enriched one
        self._openai_client = OpenAI(api_key=openai_api_key)
        self._chunker = TextChunker()

    def call(self) -> dict:
        """Entry point: prepare storage, then process every PDF in the data dir."""
        # 1) Make sure the collections and their indexes exist before writing.
        self._ensure_collections()
        self._ensure_indexes()

        # 2) Find the PDFs. If there are none, there's nothing to do.
        pdf_paths = self._list_pdfs()
        if not pdf_paths:
            print(f"No PDF files found in '{self._data_dir}'")
            return {"files": 0, "embedded": 0, "skipped": 0}

        # 3) Process the PDFs concurrently, tallying embedded vs already done.
        # Each file is fully independent, so we hand them to a bounded thread pool
        # (max_workers = _MAX_PARALLEL_FILES = our concurrency cap). We also hand
        # out a small pool of "progress-bar slots" so concurrent files draw their
        # progress bars on separate terminal lines instead of overwriting each
        # other. The tally is done here on the main thread from each future's
        # return value, so no locks are needed.
        embedded, skipped = 0, 0
        slots: Queue[int] = Queue()
        for position in range(_MAX_PARALLEL_FILES):
            slots.put(position)

        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_FILES) as executor:
            futures = [
                executor.submit(self._process_pdf, path, slots) for path in pdf_paths
            ]
            for future in as_completed(futures):
                if future.result():
                    embedded += 1
                else:
                    skipped += 1

        # 4) Print and return a small summary so callers (e.g. the Flow) can report.
        print(
            f"Done. {embedded} file(s) embedded/updated, {skipped} skipped "
            f"(already complete), {len(pdf_paths)} total."
        )
        return {"files": len(pdf_paths), "embedded": embedded, "skipped": skipped}

    def _list_pdfs(self) -> list[Path]:
        # `sorted` gives a stable, predictable processing order across runs.
        return sorted(self._data_dir.glob("*.pdf"))

    def _process_pdf(self, path: Path, slots: "Queue[int]") -> bool:
        """Embed a single PDF. Returns True if any work was done, False if skipped.

        Runs on a worker thread. ``slots`` is the shared pool of progress-bar
        positions; we borrow one for the duration of the page loop so concurrent
        files don't draw their bars on top of each other. Note we use
        ``tqdm.write`` (not ``print``) for log lines so they don't corrupt any
        live progress bars.
        """
        # The extractor wraps one PDF and exposes its name, md5, page count, and
        # lazy per-page text. We compute a few identifiers up front:
        extractor = PdfExtractor(path)
        file_md5 = extractor.file_md5()  # content fingerprint: changes if file changes
        file_name = extractor.file_name
        file_path = str(path)
        total_pages = extractor.page_count()

        # Parse the DO "issue" info (number, year, date, Extra flag) once per file.
        # It reads only the first page, falling back to the filename when needed.
        do_issue = DoIssueParser(extractor.first_page_text(), file_name).parse()

        # IDEMPOTENCY (fast path): if every page of THIS exact file version is
        # already stored in both collections, skip the whole file without opening
        # it again. This is what makes re-running the script cheap and safe.
        if self._is_file_complete(file_md5, total_pages):
            tqdm.write(f"Skipping '{file_name}' — already fully embedded")
            return False

        # If this path was previously embedded from a different version of the
        # file (same name, different md5), drop those stale chunks first so we
        # don't mix old and new content.
        self._delete_stale_versions(file_path, file_md5)

        tqdm.write(
            f"Embedding '{file_name}' ({total_pages} pages) "
            f"[issue={do_issue.issue_number}, date={do_issue.edition_date}, "
            f"extra={do_issue.is_extra}, source={do_issue.source}]"
        )
        # Borrow a progress-bar slot (its terminal line/position) for this file.
        # The pool has exactly _MAX_PARALLEL_FILES slots, matching the worker
        # count, so this get() never actually blocks; we always return it.
        position = slots.get()
        try:
            # Stream the PDF page by page (never the whole file in memory). The
            # bar is drawn at our borrowed position; leave=False clears it when
            # the file finishes so the terminal stays tidy.
            with tqdm(
                total=total_pages,
                desc=file_name,
                unit="page",
                position=position,
                leave=False,
            ) as pbar:
                for page in extractor.iter_pages():
                    self._process_page(
                        page_number=page.number,
                        page_text=page.text,
                        file_name=file_name,
                        file_path=file_path,
                        file_md5=file_md5,
                        total_pages=total_pages,
                        do_issue=do_issue,
                    )
                    pbar.update(1)
        finally:
            slots.put(position)
        return True

    def _process_page(
        self,
        page_number: int,
        page_text: str,
        file_name: str,
        file_path: str,
        file_md5: str,
        total_pages: int,
        do_issue: DoIssue,
    ) -> None:
        """Chunk, embed, and store a single page (the core of the pipeline)."""
        # Normalize whitespace so the page "fingerprint" is stable. If the page is
        # blank (e.g. a scanned image with no text layer), there is nothing to do.
        normalized = _WHITESPACE_RE.sub(" ", page_text).strip()
        if not normalized:
            return  # blank page, nothing to embed

        # Split the page into token-sized passages. Each chunk becomes one stored
        # document with its own embedding.
        chunks = self._chunker.chunk_text(page_text)
        if not chunks:
            return

        # `page_hash` uniquely identifies "this page content of this file version".
        # `expected` is how many chunks this page should produce.
        page_hash = self._page_hash(file_md5, page_number, normalized)
        expected = len(chunks)

        # IDEMPOTENCY (per page): count what's already stored for this page in each
        # collection. We identify a page's documents two ways:
        #   - enriched: by the metadata.page_hash field
        #   - plain: by the deterministic _id prefix "page_hash:" (no metadata there)
        enriched_existing = self._enriched.count_documents(
            {"metadata.page_hash": page_hash}
        )
        plain_existing = self._plain.count_documents(
            {"_id": {"$regex": f"^{re.escape(page_hash)}:"}}
        )

        # A collection is "ok" for this page when it already has exactly the number
        # of chunks we expect. If BOTH are complete, skip (no embedding cost).
        enriched_ok = enriched_existing == expected
        plain_ok = plain_existing == expected
        if enriched_ok and plain_ok:
            return  # page already fully embedded in both collections

        # We only reach the (paid) embedding call when at least one collection is
        # missing/incomplete for this page. We embed ONCE and reuse the vectors
        # for both collections so their embeddings are guaranteed identical.
        embeddings = self._embed_texts(chunks)

        # --- Write to the ENRICHED collection (text + embedding + metadata) ------
        if not enriched_ok:
            if enriched_existing:
                # Partial leftover from an interrupted run — clear and redo.
                self._enriched.delete_many({"metadata.page_hash": page_hash})
            self._enriched.insert_many(
                [
                    {
                        # Deterministic _id: same page+chunk always maps to the same
                        # document, which is what makes re-runs idempotent.
                        "_id": f"{page_hash}:{idx}",
                        "text": chunk_text,
                        "embedding": list(embedding),
                        # The metadata block is the whole point of the "enriched"
                        # collection: it lets retrieval filter/cite by source, page,
                        # and the parsed DO issue fields.
                        "metadata": {
                            "file_name": file_name,
                            "file_path": file_path,
                            "file_md5": file_md5,
                            "page": page_number,
                            "total_pages": total_pages,
                            "page_hash": page_hash,
                            "chunk_index": idx,
                            "page_chunk_count": expected,
                            "do_issue_number": do_issue.issue_number,
                            "do_year": do_issue.year,
                            "edition_date": do_issue.edition_date,
                            "is_extra": do_issue.is_extra,
                            "issue_source": do_issue.source,
                        },
                    }
                    for idx, (chunk_text, embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ]
            )

        # --- Write to the PLAIN collection (text + embedding only) ---------------
        if not plain_ok:
            if plain_existing:
                self._plain.delete_many(
                    {"_id": {"$regex": f"^{re.escape(page_hash)}:"}}
                )
            self._plain.insert_many(
                [
                    {
                        # Same deterministic _id as the enriched doc, so the two
                        # collections stay perfectly aligned per chunk.
                        "_id": f"{page_hash}:{idx}",
                        "text": chunk_text,
                        "embedding": list(embedding),
                    }
                    for idx, (chunk_text, embedding) in enumerate(
                        zip(chunks, embeddings)
                    )
                ]
            )

    def _is_file_complete(self, file_md5: str, total_pages: int) -> bool:
        """Fast path: every page of this file version is present in BOTH collections.

        The enriched collection carries page metadata, so its completeness is
        checked directly. The plain collection has no metadata, so it is verified
        by mirroring the enriched collection's per-page chunk counts via the
        shared deterministic ``_id`` (``f"{page_hash}:{idx}"``).
        """
        # First, cheap check: does the enriched collection already have a document
        # for every page number of this file version?
        enriched_pages = self._enriched.distinct(
            "metadata.page", {"metadata.file_md5": file_md5}
        )
        if len(enriched_pages) < total_pages:
            return False

        # Enriched looks complete. Now confirm the plain collection mirrors it,
        # page by page, by comparing chunk counts for each page_hash.
        page_hashes = self._enriched.distinct(
            "metadata.page_hash", {"metadata.file_md5": file_md5}
        )
        for page_hash in page_hashes:
            expected = self._enriched.count_documents({"metadata.page_hash": page_hash})
            plain_count = self._plain.count_documents(
                {"_id": {"$regex": f"^{re.escape(page_hash)}:"}}
            )
            if plain_count != expected:
                return False
        return True

    def _delete_stale_versions(self, file_path: str, file_md5: str) -> None:
        """Remove chunks for the same path but a different (older) file version."""
        # "Same file path, but NOT the current md5" => leftovers from an older
        # version of this PDF that should be replaced.
        stale_filter = {
            "metadata.file_path": file_path,
            "metadata.file_md5": {"$ne": file_md5},
        }
        # The plain collection has no metadata, but shares the enriched _ids, so
        # collect the stale ids from the enriched collection and remove them from
        # both.
        stale_ids = [
            doc["_id"] for doc in self._enriched.find(stale_filter, {"_id": 1})
        ]
        if stale_ids:
            self._plain.delete_many({"_id": {"$in": stale_ids}})
        self._enriched.delete_many(stale_filter)

    @staticmethod
    def _page_hash(file_md5: str, page_number: int, normalized_text: str) -> str:
        """Stable fingerprint of one page's content within one file version.

        Combining the file md5, the page number, and the normalized text means the
        hash changes if (and only if) the underlying content changes — which is
        exactly what we want for detecting "already embedded" pages.
        """
        key = f"{file_md5}:{page_number}:{normalized_text}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Turn a list of text chunks into a list of embedding vectors.

        We send the chunks to OpenAI in batches (default 32) rather than one call
        per chunk: fewer round-trips = faster and cheaper. The order of returned
        embeddings matches the order of the input texts.
        """
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i : i + self._embed_batch_size]
            response = self._openai_client.embeddings.create(
                model=_EMBEDDING_MODEL, input=batch
            )
            embeddings.extend(data.embedding for data in response.data)
        return embeddings

    def _ensure_collections(self) -> None:
        """Create the two collections if they don't exist yet (no-op otherwise)."""
        existing = set(self._db.list_collection_names())
        for name in (_COLLECTION_PLAIN, _COLLECTION_ENRICHED):
            if name not in existing:
                self._db.create_collection(name)

    def _ensure_indexes(self) -> None:
        """Create the btree + vector indexes needed for lookups and search."""
        # These ordinary indexes speed up the idempotency/cleanup queries above.
        # They only exist on the enriched collection because only it has metadata.
        self._enriched.create_index([("metadata.page_hash", ASCENDING)])
        self._enriched.create_index([("metadata.file_md5", ASCENDING)])
        self._enriched.create_index([("metadata.file_path", ASCENDING)])
        self._enriched.create_index([("metadata.page", ASCENDING)])

        # Both collections need the vector index so they can be queried by
        # semantic similarity at retrieval time.
        for collection in (self._plain, self._enriched):
            self._ensure_vector_index(collection)

    def _ensure_vector_index(self, collection: Collection) -> None:
        """Create the Atlas Vector Search index on a collection if it's missing."""
        existing = {idx["name"] for idx in collection.list_search_indexes()}
        if _VECTOR_INDEX_NAME in existing:
            return

        # This tells Atlas: index the `embedding` field as a 3072-dim vector and
        # compare vectors using cosine similarity. These numbers MUST match the
        # embedding model's output (see _EMBEDDING_MODEL/_EMBEDDING_DIMENSIONS).
        search_index = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "path": "embedding",
                        "type": "vector",
                        "numDimensions": _EMBEDDING_DIMENSIONS,
                        "similarity": "cosine",
                    }
                ]
            },
            name=_VECTOR_INDEX_NAME,
            type="vectorSearch",
        )
        collection.create_search_index(search_index)
        # Atlas builds the index asynchronously, so wait until it's queryable.
        self._wait_for_vector_index(collection)
        print(
            f"Created vector search index '{_VECTOR_INDEX_NAME}' on '{collection.name}'"
        )

    def _wait_for_vector_index(self, collection: Collection) -> None:
        """Poll until the vector index reports READY, or time out."""
        start = time.time()
        while time.time() - start < _VECTOR_INDEX_POLL_TIMEOUT:
            indexes = list(collection.list_search_indexes(_VECTOR_INDEX_NAME))
            if indexes and indexes[0].get("status") == "READY":
                return
            print(
                f"Waiting for vector search index on '{collection.name}' to be "
                f"ready ({int(time.time() - start)}s elapsed)..."
            )
            time.sleep(_VECTOR_INDEX_POLL_INTERVAL)

        # If the index never becomes ready, fail loudly rather than silently
        # writing into an unusable collection.
        raise TimeoutError(
            f"Vector search index on '{collection.name}' did not become ready "
            f"within {_VECTOR_INDEX_POLL_TIMEOUT}s"
        )
