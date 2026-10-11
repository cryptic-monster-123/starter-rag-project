import anthropic

from . import config
from .retrieve import Chunk, retrieve

SYSTEM = """You answer questions using only the provided context passages.
- Answer concisely, usually in one sentence or a short phrase.
- Cite the passages you used like [1], [2].
- If the context does not contain the answer, reply exactly: I don't know."""

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    ctx = "\n\n".join(f"[{n}] ({c.title})\n{c.text}" for n, c in enumerate(chunks, 1))
    return f"<context>\n{ctx}\n</context>\n\nQuestion: {question}"


def answer(question: str, k: int = config.TOP_K) -> tuple[str, list[Chunk]]:
    chunks = retrieve(question, k)
    response = client().beta.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM,
        output_config={"effort": "low"},
        # Retry on another model if the request is declined by a safety classifier.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
    )
    if response.stop_reason == "refusal":
        return "The model declined to answer this question.", chunks
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text, chunks
