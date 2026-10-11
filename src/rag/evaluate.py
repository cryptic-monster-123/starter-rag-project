"""Evaluate on SQuAD v2 questions.

Metrics:
- retrieval hit@k: gold passage is in the top-k (answerable questions only). No API calls.
- answer accuracy (--generate): a gold answer string appears in Claude's reply for
  answerable questions; reply is "I don't know" for unanswerable ones.

Usage:
    python -m rag.evaluate --n 200                 # retrieval only, free
    python -m rag.evaluate --n 50 --generate       # also calls Claude
"""
import argparse
import json
import random

from datasets import load_dataset

from .generate import answer
from .ingest import passage_id
from .retrieve import retrieve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/results.jsonl")
    args = p.parse_args()

    ds = load_dataset("rajpurkar/squad_v2", split=args.split)
    rows = random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds)))

    hits = answerable = correct = 0
    with open(args.out, "w") as f:
        for idx in rows:
            row = ds[idx]
            gold = row["answers"]["text"]
            gold_pid = passage_id(row["context"])
            rec = {"question": row["question"], "gold": gold}

            if args.generate:
                reply, chunks = answer(row["question"], args.k)
                if gold:
                    ok = any(g.lower() in reply.lower() for g in gold)
                else:
                    ok = "i don't know" in reply.lower()
                correct += ok
                rec.update(reply=reply, correct=ok)
            else:
                chunks = retrieve(row["question"], args.k)

            if gold:
                answerable += 1
                hit = gold_pid in [c.id for c in chunks]
                hits += hit
                rec["hit"] = hit
            f.write(json.dumps(rec) + "\n")

    print(f"Questions: {len(rows)} ({answerable} answerable)")
    if answerable:
        print(f"Retrieval hit@{args.k}: {hits / answerable:.1%}")
    if args.generate:
        print(f"Answer accuracy: {correct / len(rows):.1%}")
    print(f"Details: {args.out}")


if __name__ == "__main__":
    main()
