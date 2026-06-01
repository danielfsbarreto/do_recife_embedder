# do_recife_embedder

Embeds *Diário Oficial do Recife* PDFs into MongoDB Atlas Vector Search for use in a RAG pipeline.

Each PDF is streamed page by page, split into token-sized chunks, embedded once with OpenAI, and written to **two** collections that share identical text and vectors but differ in stored shape:

- `do-recife-rag` — plain: `{ _id, text, embedding }` only.
- `do-recife-rag-enriched` — same, plus a `metadata` object (source file, page, and parsed DO issue: number, year, edition date, extra flag).

The two collections let you compare retrieval quality with vs. without metadata.

## Requirements

- Python >=3.10, <3.14
- [uv](https://docs.astral.sh/uv/)
- A MongoDB Atlas cluster (the `vector_index` is created automatically)
- An OpenAI API key

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` (see `.env.example`):

```bash
OPENAI_API_KEY=sk-...
MONGODB_CONNECTION_STRING=mongodb+srv://...
```

The database (`do-recife`) and collection names are fixed in
`src/do_recife_embedder/services/pdf_embedding_service.py`.

## Usage

Drop your PDFs into `data/`, then run:

```bash
crewai run
```

This kicks off the flow in `src/do_recife_embedder/main.py`, which embeds every
`*.pdf` in `data/` and prints a summary of how many files were embedded vs.
skipped.

## How it works

```
data/*.pdf
  -> PdfExtractor      (stream text page by page)
  -> TextChunker       (token-sized chunks, semchunk + tiktoken)
  -> OpenAI            (text-embedding-3-large, 3072-dim, cosine)
  -> MongoDB Atlas     (do-recife-rag + do-recife-rag-enriched)
```

- **Resumable / idempotent.** Each chunk's `_id` is deterministic
  (`"{page_hash}:{chunk_index}"`, where `page_hash` is an md5 of the file md5 +
  page number + page text). Re-running skips pages already present in both
  collections and never re-embeds them.
- **Parallel.** Files are processed by a bounded thread pool; tune
  `_MAX_PARALLEL_FILES` (default `3`) in `pdf_embedding_service.py`.
- **DO issue parsing.** `DoIssueParser` reads the first page (e.g. `ANO LV - Nº
  053` and the date line), falling back to the filename
  (`DO Recife 053 Edição 05-05-2026.pdf`) for any missing field and the `Extra`
  flag.

## Project layout

```
src/do_recife_embedder/
├── main.py                          # CrewAI Flow entry point
├── services/pdf_embedding_service.py# extract -> chunk -> embed -> store
└── utils/
    ├── pdf_extractor.py             # streaming PyMuPDF text extraction
    ├── text_chunker.py              # semchunk + tiktoken chunking
    └── do_issue_parser.py           # parse DO issue from page/filename
```
