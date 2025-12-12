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

# for VG
# from pydantic import BaseModel  # <- already imported above, keep original but comment duplicate
from typing import Literal, List, Dict
from uuid import uuid4

# ---------------- Task 4: memory support helper ----------------

# OLD placeholder:
# async def generate_rag_reply(conversation_text: str) -> str:
#     # your existing Gemini + RAG call here
#     ...
#
# NEW implementation using the existing Agent + RAG
async def generate_rag_reply(conversation_text: str) -> str:
    """
    Use the existing RAG agent to generate a reply given the full conversation text.
    """
    deps = get_deps()
    result = await agent.run(conversation_text, deps=deps)
    return result.output


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


@app.get("/")
async def root():
    return {"status": "ok", "message": "App is running"}

# your other routes:
# @app.get("/something")
# @app.post("/something-else")


# Allow local Streamlit to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Task 4: Chat models with memory ----------

# OLD simple models (Task 2):
# class ChatRequest(BaseModel):
#     message: str
#     # optionally later: session_id: str, history: list, etc.
#
# class ChatResponse(BaseModel):
#     reply: str

# NEW models with memory + history (Task 4 / VG)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    history: List[ChatMessage]


# In-memory store of histories (per session_id)
histories: Dict[str, List[ChatMessage]] = {}


# ---------- Task 4: Chat endpoint with memory ----------

# OLD Task 2 chat endpoint:
# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(req: ChatRequest) -> ChatResponse:
#     deps = get_deps()
#     try:
#         result = await agent.run(req.message, deps=deps)
#         reply = result.output
#         return ChatResponse(reply=reply)
#     except Exception as e:
#         print("❌ Error in /chat:", repr(e))  # This will show full error in the terminal
#         raise HTTPException(
#             status_code=500,
#             detail="Internal error in chat backend"
#         )

# OLD duplicated chat for VG (broken because it redefines ChatRequest/Response below):
# @app.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest):
#     session_id = req.session_id
#     user_message = req.message
#
#     # Get or create history for this session
#     history = histories.setdefault(session_id, [])
#     history.append(ChatMessage(role="user", content=user_message))
#
#     # Build context for RAG: combine history messages
#     conversation_text = "\n".join(
#         f"{m.role}: {m.content}" for m in history
#     )
#
#     # Use your existing RAG logic but pass conversation_text as context
#     reply_text = await generate_rag_reply(conversation_text)
#
#     # Store assistant reply
#     history.append(ChatMessage(role="assistant", content=reply_text))
#
#     return ChatResponse(reply=reply_text, history=history)
# #till here

# NEW single /chat endpoint combining RAG + memory

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    user_message = req.message

    # Get or create history for this session
    history = histories.setdefault(session_id, [])
    history.append(ChatMessage(role="user", content=user_message))

    # Build context for RAG: combine history messages
    conversation_text = "\n".join(
        f"{m.role}: {m.content}" for m in history
    )

    try:
        # Use existing RAG logic with full conversation as input
        reply_text = await generate_rag_reply(conversation_text)
    except Exception as e:
        print("❌ Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail="Internal error in chat backend")

    # Store assistant reply
    history.append(ChatMessage(role="assistant", content=reply_text))

    return ChatResponse(reply=reply_text, history=history)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Task 4: History endpoint ----------

@app.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_history(session_id: str):
    """
    Returns the stored chat history for a given session_id.
    """
    return histories.get(session_id, [])


# ---------- Task 4: Video description & tags endpoints ----------

class VideoRequest(BaseModel):
    video_id: str


class DescriptionResponse(BaseModel):
    video_id: str
    description: str


class TagsResponse(BaseModel):
    video_id: str
    tags: str  # comma-separated list: keyword1,keyword2,...


@app.post("/video/description", response_model=DescriptionResponse)
async def video_description(req: VideoRequest) -> DescriptionResponse:
    """
    Generate a YouTube description for a given video_id using the existing RAG agent.
    """
    video_id = req.video_id

    prompt = f"""
You are The Youtuber. Based on my course content and transcripts, write a YouTube
description for the video with id `{video_id}`.

Requirements:
- Conversational and in my teaching style.
- 3–6 short paragraphs.
- Explain what the viewer will learn.
- Mention that this is part of my course.
- Include a short call to action (like/subscribe/check the course).
- Do NOT include hashtags.
"""

    deps = get_deps()
    try:
        result = await agent.run(prompt, deps=deps)
        description = result.output.strip()
    except Exception as e:
        print("❌ Error in /video/description:", repr(e))
        raise HTTPException(status_code=500, detail="Failed to generate description")

    return DescriptionResponse(video_id=video_id, description=description)


@app.post("/video/tags", response_model=TagsResponse)
async def video_tags(req: VideoRequest) -> TagsResponse:
    """
    Generate 20–40 keywords (tags) for a given video_id.
    Output format: keyword1,keyword2,keyword3,...
    """
    video_id = req.video_id

    prompt = f"""
You are The Youtuber’s assistant.
Based on my course content and transcripts, generate 20–40 SEO-friendly keywords
for the video with id `{video_id}` that I can use as YouTube tags.

Rules:
- Return ONLY a comma-separated list of tags.
- No numbering, no extra words, no explanations.
- No spaces after commas (exact format: keyword1,keyword2,keyword3,...).
- Each tag should be a short phrase (1–3 words).
"""

    deps = get_deps()
    try:
        result = await agent.run(prompt, deps=deps)
        raw = result.output.strip()
    except Exception as e:
        print("❌ Error in /video/tags:", repr(e))
        raise HTTPException(status_code=500, detail="Failed to generate tags")

    # Basic cleanup to enforce comma-separated format without spaces
    # Convert newlines to commas, split, strip, re-join
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    cleaned = ",".join(parts)

    return TagsResponse(video_id=video_id, tags=cleaned)

# OLD duplicated models at bottom (now superseded by Task 4 models above)
# # for VG
#
# class ChatMessage(BaseModel):
#     role: Literal["user", "assistant"]
#     content: str
#
# class ChatRequest(BaseModel):
#     session_id: str
#     message: str
#
# class ChatResponse(BaseModel):
#     reply: str
#     history: List[ChatMessage]
