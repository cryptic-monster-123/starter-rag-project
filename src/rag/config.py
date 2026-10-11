import os

from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma")
COLLECTION = os.getenv("COLLECTION", "squad_v2")
TOP_K = int(os.getenv("TOP_K", "5"))
