"""Chroma collection with a local sentence-transformers embedding function."""
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from . import config


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    return client.get_or_create_collection(
        name=config.COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
