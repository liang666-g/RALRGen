# -*- coding: utf-8 -*-
"""Extract aligned records using the 1-based sentiment index files."""

import argparse
import json
from pathlib import Path


LABELS = ("Positive", "Neutral", "Negative")


def read_lines(path):
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        return [line.rstrip("\r\n") for line in handle]


def read_indices(path, source_size):
    indices = []
    for index_line, value in enumerate(read_lines(path), start=1):
        try:
            index = int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{path}, line {index_line}: invalid index {value!r}"
            ) from exc
        if not 1 <= index <= source_size:
            raise ValueError(
                f"{path}, line {index_line}: index {index} is outside "
                f"the source range 1..{source_size}"
            )
        indices.append(index)
    return indices


def extract(source_path, index_dir, output_dir):
    source_path = Path(source_path).resolve()
    index_dir = Path(index_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_lines = read_lines(source_path)
    all_indices = []
    counts = {}
    output_paths = {}

    for label in LABELS:
        index_path = index_dir / f"{label}-indices.txt"
        indices = read_indices(index_path, len(source_lines))
        output_path = output_dir / f"{label}.txt"
        selected = [source_lines[index - 1] for index in indices]
        output_path.write_text(
            "".join(line + "\n" for line in selected), encoding="utf-8"
        )

        all_indices.extend(indices)
        counts[label] = len(indices)
        output_paths[label] = str(output_path)

    expected = list(range(1, len(source_lines) + 1))
    if sorted(all_indices) != expected:
        raise ValueError(
            "The sentiment indices do not form a complete, unique partition "
            "of the source records"
        )

    manifest = {
        "schema_version": 1,
        "source_path": str(source_path),
        "source_records": len(source_lines),
        "index_dir": str(index_dir),
        "index_numbering": "1-based",
        "class_counts": counts,
        "complete_unique_partition": True,
        "files": output_paths,
    }
    manifest_path = output_dir / "llm_sentiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Source: {source_path}")
    print(f"Source records: {len(source_lines)}")
    for label in LABELS:
        print(f"{label}: {counts[label]}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract records using sentiment classification indices."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--index-dir", default="./data/sentiment")
    parser.add_argument("--output-dir", default="./data/llm_sentiment")
    args = parser.parse_args()
    extract(args.source, args.index_dir, args.output_dir)


if __name__ == "__main__":
    main()
