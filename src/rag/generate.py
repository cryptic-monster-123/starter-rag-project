import ollama

from . import config
from .retrieve import Chunk, retrieve

SYSTEM = """You answer questions using only the provided context passages.
- Answer concisely, usually in one sentence or a short phrase.
- Cite the passages you used like [1], [2].
- If the context does not contain the answer, reply exactly: I don't know."""

_client = None


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=config.OLLAMA_HOST)
    return _client


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n".join(f"[{n}] ({c.title})\n{c.text}" for n, c in enumerate(chunks, 1))
    return f"<context>\n{ctx}\n</context>\n\nQuestion: {question}"


def answer(question: str, k: int = config.TOP_K) -> tuple[str, list[Chunk]]:
    chunks = retrieve(question, k)
    response = client().chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        options={"temperature": 0, "num_ctx": 8192},
    )
    return response.message.content.strip(), chunks
