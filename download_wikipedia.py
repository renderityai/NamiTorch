# pip install datasets
from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.pl",
    split="train",
    streaming=True
)

limit = 100 * 1024 * 1024
written = 0

with open("data/corpus.txt", "w", encoding="utf-8") as f:
    for row in dataset:
        text = row["text"].strip()

        if not text:
            continue

        chunk = text + "\n\n"

        f.write(chunk)

        written += len(chunk.encode("utf-8"))

        print(f"\r{written / 1024 / 1024:.2f} MB", end="")

        if written >= limit:
            break

print("\nGotowe.")