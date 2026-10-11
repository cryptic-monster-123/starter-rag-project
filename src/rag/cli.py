"""Ask a question from the terminal: python -m rag.cli "When did Beyonce start becoming popular?\""""
import sys

from .generate import answer


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")
    text, chunks = answer(question)
    print(f"\n{text}\n\nSources:")
    for n, c in enumerate(chunks, 1):
        print(f"  [{n}] {c.title} (score {c.score:.2f}): {c.text[:100]}...")


if __name__ == "__main__":
    main()
