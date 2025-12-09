# src/ingestion.py
"""
Task 0 - Data ingestion into LanceDB (using Gemini embeddings).

LLM-assisted implementation: I've used an AI for scaffolding this file
and then reviewed/modified it myself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field

import google.generativeai as genai

from .config import LANCEDB_URI, GEMINI_API_KEY, BASE_DIR


# ----- 1. Define the LanceDB row model -----


class TranscriptChunk(LanceModel):
    id: str
    video_id: str
    chunk_index: int
    text: str
    # Gemini text-embedding-004 -> 768-dimensional vector
    vector: Vector(768) = Field(description="Embedding vector for this chunk")


# ----- 2. Embedding client wrapper (Gemini) -----


@dataclass
class EmbeddingClient:
    # Gemini embedding model
    model: str = "models/text-embedding-004"

    def __post_init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Set it in your environment or .env file."
            )
        genai.configure(api_key=GEMINI_API_KEY)

    def embed(self, text: str) -> List[float]:
        """
        Call Gemini's embeddings API and return a list[float].
        """
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        resp = genai.embed_content(
            model=self.model,
            content=text,
            # This param is optional, 768 is the default for text-embedding-004,
            # but we make it explicit to match Vector(768)
            task_type="SEMANTIC_SIMILARITY",
        )
        # embed_content returns a dict with "embedding"
        return resp["embedding"]


# ----- 3. Utility: load all text files under data/ -----


def load_transcripts(data_dir: Path) -> list[tuple[str, str]]:
    """
    Loads all transcript-like files from the data directory.

    Looks for:
      - *.txt
      - *.md

    Returns:
        list of (video_id, full_text)
    """
    transcripts: list[tuple[str, str]] = []

    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return transcripts

    # Recursively grab .txt and .md files under data/
    txt_files = list(data_dir.rglob("*.txt"))
    md_files = list(data_dir.rglob("*.md"))
    all_files = sorted(txt_files + md_files)

    if not all_files:
        print(f"No .txt or .md files found under {data_dir}")
        return transcripts

    print(f"Found {len(all_files)} text files under {data_dir}:")
    for path in all_files:
        print("  -", path)

    for path in all_files:
        video_id = path.stem  # e.g. "data_storytelling" or "video1_transcript"
        text = path.read_text(encoding="utf-8")
        transcripts.append((video_id, text))

    return transcripts


# ----- 4. Chunking function -----


def chunk_text(text: str, max_tokens: int = 300) -> list[str]:
    """
    Very simple chunking: split on words and group into chunks
    of ~max_tokens words.
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# ----- 5. Main ingestion flow -----


def ingest_transcripts():
    # 5.1 Open LanceDB connection
    db = lancedb.connect(LANCEDB_URI)

    # 5.2 Get or create table
    table_name = "transcript_chunks"
    if table_name in db.table_names():
        table = db.open_table(table_name)
    else:
        table = db.create_table(
            table_name,
            schema=TranscriptChunk,
        )

    embedder = EmbeddingClient()

    # Use BASE_DIR / "data" (all your .md/.txt live here)
    data_dir = BASE_DIR / "data"
    all_transcripts = load_transcripts(data_dir)

    rows: list[TranscriptChunk] = []

    for video_id, full_text in all_transcripts:
        chunks = chunk_text(full_text, max_tokens=300)
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            try:
                embedding = embedder.embed(chunk)
            except Exception as e:
                print(f"[WARN] Failed to embed chunk {video_id}_{idx}: {e}")
                continue

            row = TranscriptChunk(
                id=f"{video_id}_{idx}",
                video_id=video_id,
                chunk_index=idx,
                text=chunk,
                vector=embedding,
            )
            rows.append(row)

    if not rows:
        print("No transcript chunks found. Make sure there are .txt/.md files under the data/ folder.")
        return

    # 5.3 Add data to the table
    table.add(rows)
    print(f"Ingested {len(rows)} chunks into LanceDB table '{table_name}'.")


if __name__ == "__main__":
    ingest_transcripts()