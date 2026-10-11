"""Load SQuAD v2 from Hugging Face and index its unique passages into Chroma.

Usage:
    python -m rag.ingest --split validation --limit 0
"""
import argparse
import hashlib

from datasets import load_dataset
from tqdm import tqdm

from .store import get_collection


def passage_id(context: str) -> str:
    return hashlib.sha1(context.encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", help="train (~19k passages) or validation (~1.2k)")
    p.add_argument("--limit", type=int, default=0, help="max passages to index (0 = all)")
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    ds = load_dataset("rajpurkar/squad_v2", split=args.split)

    # SQuAD repeats each passage for every question; keep one copy.
    passages = {}
    for row in ds:
        pid = passage_id(row["context"])
        if pid not in passages:
            passages[pid] = {"text": row["context"], "title": row["title"]}
            if args.limit and len(passages) >= args.limit:
                break

    col = get_collection()
    ids = list(passages)
    for i in tqdm(range(0, len(ids), args.batch), desc="Indexing"):
        batch = ids[i : i + args.batch]
        col.upsert(
            ids=batch,
            documents=[passages[b]["text"] for b in batch],
            metadatas=[{"title": passages[b]["title"]} for b in batch],
        )
    print(f"Indexed {len(ids)} passages. Collection size: {col.count()}")


if __name__ == "__main__":
    main()
