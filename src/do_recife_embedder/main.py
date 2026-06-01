#!/usr/bin/env python
from crewai.flow import Flow, listen, start
from pydantic import BaseModel

from do_recife_embedder.services import PdfEmbeddingService


class EmbeddingState(BaseModel):
    data_dir: str = "data"
    files: int = 0
    embedded: int = 0
    skipped: int = 0


class PdfEmbeddingFlow(Flow[EmbeddingState]):
    @start()
    def embed_pdfs(self):
        print(f"Embedding PDFs from: {self.state.data_dir}")
        result = PdfEmbeddingService(data_dir=self.state.data_dir).call()
        self.state.files = result["files"]
        self.state.embedded = result["embedded"]
        self.state.skipped = result["skipped"]

    @listen(embed_pdfs)
    def report(self):
        print(
            f"Embedding finished: {self.state.embedded} embedded/updated, "
            f"{self.state.skipped} skipped of {self.state.files} PDF(s)."
        )
        return self.state.model_dump()


def kickoff():
    PdfEmbeddingFlow().kickoff()


def plot():
    PdfEmbeddingFlow().plot()


if __name__ == "__main__":
    kickoff()
