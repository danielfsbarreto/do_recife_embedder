import re
import unicodedata
from dataclasses import dataclass


@dataclass
class DoIssue:
    """Structured issue information for a Diário Oficial do Recife edition."""

    issue_number: str | None  # e.g. "053"
    year: str | None  # e.g. "LV" (the "ANO" roman numeral)
    edition_date: str | None  # ISO date, e.g. "2026-05-05"
    is_extra: bool
    source: str  # "document" | "filename" | "mixed" | "none"


_PT_MONTHS = {
    "JANEIRO": 1,
    "FEVEREIRO": 2,
    "MARCO": 3,
    "ABRIL": 4,
    "MAIO": 5,
    "JUNHO": 6,
    "JULHO": 7,
    "AGOSTO": 8,
    "SETEMBRO": 9,
    "OUTUBRO": 10,
    "NOVEMBRO": 11,
    "DEZEMBRO": 12,
}

_DOC_ISSUE_RE = re.compile(r"N[ºo°˚]\s*0*(\d{1,4})", re.IGNORECASE)
_DOC_YEAR_RE = re.compile(r"\bANO\s+([IVXLCDM]+)\b", re.IGNORECASE)
_DOC_DATE_RE = re.compile(
    r"RECIFE,.*?(\d{1,2})\s+DE\s+([A-ZÇÃÉÍÁÊÓÔÕ]+)\s+DE\s+(\d{4})",
    re.IGNORECASE | re.DOTALL,
)
# Matched against the accent-stripped filename, so "Edição" becomes "Edicao".
# This avoids NFC/NFD mismatches (macOS exposes filenames in decomposed form).
_FILENAME_RE = re.compile(
    r"DO\s+Recife\s+(\d+)\s+Edicao\s+(\d{2})-(\d{2})-(\d{4})(?:\s+(Extra))?",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


class DoIssueParser:
    """Parse a DO edition's issue info from the first page, then the filename.

    Document text is preferred; any field the document does not yield falls back
    to the filename (which reliably carries issue number, edition date, and the
    ``Extra`` flag).
    """

    def __init__(self, first_page_text: str | None, file_name: str):
        self._first_page_text = first_page_text or ""
        self._file_name = file_name or ""

    def parse(self) -> DoIssue:
        doc = self._parse_document()
        filename = self._parse_filename()

        used_doc = False
        used_filename = False

        issue_number = doc.get("issue_number")
        if issue_number:
            used_doc = True
        elif filename.get("issue_number"):
            issue_number = filename["issue_number"]
            used_filename = True

        year = doc.get("year")
        if year:
            used_doc = True

        edition_date = doc.get("edition_date")
        if edition_date:
            used_doc = True
        elif filename.get("edition_date"):
            edition_date = filename["edition_date"]
            used_filename = True

        # The Extra flag is authoritative from the filename.
        is_extra = filename.get("is_extra", False)

        if used_doc and used_filename:
            source = "mixed"
        elif used_doc:
            source = "document"
        elif used_filename or is_extra:
            source = "filename"
        else:
            source = "none"

        return DoIssue(
            issue_number=issue_number,
            year=year,
            edition_date=edition_date,
            is_extra=is_extra,
            source=source,
        )

    def _parse_document(self) -> dict:
        text = self._first_page_text
        result: dict = {}

        issue_match = _DOC_ISSUE_RE.search(text)
        if issue_match:
            result["issue_number"] = issue_match.group(1).zfill(3)

        year_match = _DOC_YEAR_RE.search(text)
        if year_match:
            result["year"] = year_match.group(1).upper()

        date_match = _DOC_DATE_RE.search(text)
        if date_match:
            day = int(date_match.group(1))
            month_name = _strip_accents(date_match.group(2)).upper()
            year = int(date_match.group(3))
            month = _PT_MONTHS.get(month_name)
            if month:
                result["edition_date"] = f"{year:04d}-{month:02d}-{day:02d}"

        return result

    def _parse_filename(self) -> dict:
        match = _FILENAME_RE.search(_strip_accents(self._file_name))
        if not match:
            return {}

        issue_number, day, month, year, extra = match.groups()
        return {
            "issue_number": issue_number.zfill(3),
            "edition_date": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
            "is_extra": extra is not None,
        }
