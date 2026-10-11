from dataclasses import dataclass

from . import config
from .store import get_collection


@dataclass
class Chunk:
    id: str
    text: str
    title: str
    score: float  # cosine similarity (1 = identical)


def retrieve(query: str, k: int = config.TOP_K) -> list[Chunk]:
    res = get_collection().query(query_texts=[query], n_results=k)
    return [
        Chunk(id=i, text=d, title=m.get("title", ""), score=1 - dist)
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
