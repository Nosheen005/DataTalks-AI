# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env (if present)
load_dotenv(BASE_DIR / ".env")

# --- API Keys ---
# Gemini API key (required for LLM)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY is not set. Gemini LLM will not work.")

# --- LanceDB storage ---
LANCEDB_URI = os.getenv("LANCEDB_URI", str(BASE_DIR / "lancedb"))
