# src/api.py
"""
Task 2 - FastAPI API to serve the RAG chatbot.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import lancedb
from pydantic_ai import Agent, RunContext, Tool

from lancedb.pydantic import LanceModel, Vector



from .config import LANCEDB_URI
from .ingestion import EmbeddingClient, TranscriptChunk   # reuse Task 0/1
from fastapi import HTTPException


# ---------- 1. RAG deps & setup (same logic as rag_app.py) ----------

@dataclass
class RAGDeps:
    table: any
    embedder: EmbeddingClient


def get_deps() -> RAGDeps:
    db = lancedb.connect(LANCEDB_URI)
    table = db.open_table("transcript_chunks")
    embedder = EmbeddingClient()
    return RAGDeps(table=table, embedder=embedder)


class RetrievedChunk(BaseModel):
    video_id: str
    chunk_index: int
    text: str


async def search_knowledge(
    ctx: RunContext[RAGDeps],
    query: str,
) -> list[RetrievedChunk]:
    deps = ctx.deps

    query_embedding = deps.embedder.embed(query)

    results = (
        deps.table
        .search(query_embedding, vector_column_name="vector")
        .limit(5)
        .to_pydantic(TranscriptChunk)
    )

    return [
        RetrievedChunk(
            video_id=row.video_id,
            chunk_index=row.chunk_index,
            text=row.text,
        )
        for row in results
    ]


search_knowledge_tool = Tool(
    search_knowledge,
    takes_ctx=True,
    name="search_knowledge",
    description=(
        "Searches the YouTuber transcript knowledge base for relevant passages. "
        "Use this whenever you need factual details from the videos."
    ),
)

SYSTEM_PROMPT = """
You are an AI assistant that embodies the personality of "The Youtuber".

Personality:
- Friendly, energetic, and down-to-earth.
- Explains concepts clearly with concrete, practical examples.
- Talks like a helpful YouTube teacher, not like a formal academic paper.
- If something is not in the transcripts, be honest and say you don't know
  rather than making things up.

Context:
- You have access to a tool called `search_knowledge` which searches
  chunks of transcripts from the Youtuber's videos stored in LanceDB.

Guidelines:
- For any question that might depend on the video content or technical details,
  you should call `search_knowledge` at least once before answering.
- When you answer, weave in the information from the retrieved chunks,
  but do not just copy them verbatim; paraphrase and explain them
  in your own friendly YouTuber style.
- If multiple chunks disagree or are unclear, mention the uncertainty
  and explain how you interpret it.
- If the user is just doing small talk or asking about you as an assistant,
  you may answer without calling the tool.
- Keep answers concise but helpful; use bullet points or short paragraphs
  when it makes explanations easier to follow.
"""

# IMPORTANT:
# Use whatever model string is currently working in your rag_app.py.
# Example if you're using Gemini via pydantic-ai:
# MODEL_ID = "google-gla:gemini-1.5-flash"
# Or for Ollama:
# MODEL_ID = "ollama:llama3.2:3b"
#MODEL_ID = "google-gla:gemini-1.5-flash"  # <-- match what you have working
#MODEL_ID = "google-gla:gemini-2.0-flash"
MODEL_ID = "google-gla:gemini-2.5-flash"



agent = Agent(
    model=MODEL_ID,
    system_prompt=SYSTEM_PROMPT,
    deps_type=RAGDeps,
    tools=[search_knowledge_tool],
)


# ---------- 2. FastAPI app ----------

app = FastAPI(title="Youtuber RAG API")

# Allow local Streamlit to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # optionally later: session_id: str, history: list, etc.


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    deps = get_deps()
    try:
        result = await agent.run(req.message, deps=deps)
        reply = result.output
        return ChatResponse(reply=reply)
    except Exception as e:
        print("❌ Error in /chat:", repr(e))  # This will show full error in the terminal
        raise HTTPException(
            status_code=500,
            detail="Internal error in chat backend"
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
