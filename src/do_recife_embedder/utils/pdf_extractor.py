import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pymupdf


@dataclass
class PdfPage:
    """A single extracted page from a PDF."""

    number: int  # 1-indexed page number
    text: str


class PdfExtractor:
    """Streaming, page-by-page PDF text extraction using base PyMuPDF.

    Only the standard PyMuPDF API is used (``import pymupdf``); the Pro module
    (``pymupdf.pro``) is never imported, so the 3-page evaluation limit does not
    apply. Pages are yielded lazily so very large PDFs are never fully buffered.
    """

    _MD5_READ_CHUNK = 1024 * 1024  # 1 MiB

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self._file_md5: str | None = None
        self._page_count: int | None = None

    @property
    def file_name(self) -> str:
        return self.file_path.name

    def file_md5(self) -> str:
        """md5 of the raw file bytes, read in chunks to stay memory-friendly."""
        if self._file_md5 is None:
            hasher = hashlib.md5()
            with self.file_path.open("rb") as f:
                for block in iter(lambda: f.read(self._MD5_READ_CHUNK), b""):
                    hasher.update(block)
            self._file_md5 = hasher.hexdigest()
        return self._file_md5

    def page_count(self) -> int:
        if self._page_count is None:
            with pymupdf.open(self.file_path) as doc:
                self._page_count = doc.page_count
        return self._page_count

    def first_page_text(self) -> str:
        """Return the text of the first page only (empty string if no pages)."""
        with pymupdf.open(self.file_path) as doc:
            self._page_count = doc.page_count
            if doc.page_count == 0:
                return ""
            return doc[0].get_text()

    def iter_pages(self) -> Iterator[PdfPage]:
        """Yield each page's text lazily (1-indexed page numbers)."""
        with pymupdf.open(self.file_path) as doc:
            self._page_count = doc.page_count
            for index, page in enumerate(doc):
                yield PdfPage(number=index + 1, text=page.get_text())
