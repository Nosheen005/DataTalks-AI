# src/rag_app.py
"""
Task 1 - RAG agent using PydanticAI + LanceDB.

LLM-assisted implementation: scaffold generated with an AI and then
reviewed/understood/adjusted by me.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool

import lancedb

from .config import LANCEDB_URI
from .ingestion import EmbeddingClient, TranscriptChunk  # reuse from Task 0


# --- 1. Dependencies object for the agent ---


@dataclass
class RAGDeps:
    """Shared dependencies for the RAG agent (db, table, embedder)."""
    table: any
    embedder: EmbeddingClient


def get_deps() -> RAGDeps:
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table("transcript_chunks")
    embedder = EmbeddingClient()
    return RAGDeps(table=table, embedder=embedder)


# --- 2. Tool: search LanceDB using embeddings ---


class RetrievedChunk(BaseModel):
    """
    Simple structure for what we return to the model.
    This avoids dumping the whole LanceModel back to the LLM.
    """
    video_id: str
    chunk_index: int
    text: str


# IMPORTANT: ctx (RunContext) must be the FIRST parameter for takes_ctx=True
async def search_knowledge(
    ctx: RunContext[RAGDeps],
    query: str,
) -> list[RetrievedChunk]:
    """
    Given a natural language query, perform a vector search in LanceDB.
    """
    deps = ctx.deps

    # 1) Embed the query using the same embedder as ingestion
    query_embedding = deps.embedder.embed(query)

    # 2) Perform similarity search against the transcript_chunks table
    results = (
        deps.table
        .search(query_embedding, vector_column_name="vector")
        .limit(5)
        .to_pydantic(TranscriptChunk)
    )

    # 3) Map to compact objects with only the fields the LLM needs
    chunks: list[RetrievedChunk] = []
    for row in results:
        chunks.append(
            RetrievedChunk(
                video_id=row.video_id,
                chunk_index=row.chunk_index,
                text=row.text,
            )
        )

    return chunks


search_knowledge_tool = Tool(
    search_knowledge,  # function first, no "fn=" kwarg
    takes_ctx=True,
    name="search_knowledge",
    description=(
        "Searches the Youtuber transcript knowledge base for relevant passages. "
        "Use this whenever you need factual details from the videos."
    ),
)


# --- 3. Define the PydanticAI agent ---


SYSTEM_PROMPT = """
You are an AI assistant that embodies the personality of "The Youtuber":

- Friendly, energetic, down-to-earth.
- Explains concepts clearly with examples.
- Avoids being too formal or academic.
- If something is not in the transcripts, be honest and say you don't know.

You have access to the function `search_knowledge` which searches
chunks of transcripts from the Youtuber's videos.

Guidelines:
- Always call `search_knowledge` at least once before answering
  any question that might depend on the video content.
- When you answer, weave in the information from the retrieved chunks,
  but do not just copy them verbatim; paraphrase and explain.
- If multiple chunks disagree or are unclear, mention the uncertainty.
- If the question is purely about your own behavior or small talk,
  you may answer without calling the tool.
"""

# Use an Ollama model; make sure it's pulled in Ollama
# e.g. `ollama pull gpt-oss:20b`
#MODEL_ID = "google:gemini-1.5-flash"
MODEL_ID = "google-gla:gemini-2.5-flash" 
# or: MODEL_ID = "google:gemini-1.5-pro"


agent = Agent(
    model=MODEL_ID,
    system_prompt=SYSTEM_PROMPT,
    deps_type=RAGDeps,
    tools=[search_knowledge_tool],
)


# --- 4. Small CLI to interact with the agent ---


def chat_loop():
    """
    Simple terminal chat to test the RAG agent.
    """
    deps = get_deps()
    print("Youtuber RAG assistant. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            print("Bye! 👋")
            break

        try:
            # Run the agent synchronously for convenience
            result = agent.run_sync(user_msg, deps=deps)
            # In pydantic-ai v1.x, the final text answer is in result.output
            print(f"\nYoutuber: {result.output}\n")
        except Exception as e:
            print(f"[Error] {e}")



if __name__ == "__main__":
    chat_loop()
