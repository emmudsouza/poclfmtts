from __future__ import annotations

import argparse
import glob
import pathlib
import re


def sentences(text: str):
    for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        yield s.strip()


def keep(s: str, min_words: int, max_words: int) -> bool:
    words = s.split()
    return (
        min_words <= len(words) <= max_words
        and s.isascii()
        and s[:1].isupper()
        and s[-1] in ".!?"
        and not any(c in s for c in "=@|<>{}[]\\")
    )


def from_files(patterns, min_words, max_words):
    for pattern in patterns:
        for path in glob.glob(pattern):
            for s in sentences(pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")):
                if keep(s, min_words, max_words):
                    yield s


def from_hf(dataset, config, column, split, min_words, max_words):
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    for row in ds:
        for s in sentences(str(row.get(column, ""))):
            if keep(s, min_words, max_words):
                yield s


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a sentence corpus (one per line) for distillation.")
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--from-files", nargs="*", help="glob(s) of local .txt files (no download)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--column", default="text")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--max-words", type=int, default=28)
    args = ap.parse_args()

    source = (from_files(args.from_files, args.min_words, args.max_words) if args.from_files
              else from_hf(args.dataset, args.config, args.column, args.split,
                           args.min_words, args.max_words))

    seen, out = set(), []
    for s in source:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) % 5000 == 0:
            print(f"  collected {len(out)}/{args.n}", flush=True)
        if len(out) >= args.n:
            break

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {len(out)} sentences to {path}")


if __name__ == "__main__":
    main()
