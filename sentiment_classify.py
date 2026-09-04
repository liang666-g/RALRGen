import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


FIELD_SEP = "-[split]-"
SENTIMENT_INDEX = 7
LABELS = ("Positive", "Neutral", "Negative")


def classify_sentiment(sentiment_pair):
    fields = sentiment_pair.split()
    if len(fields) != 2:
        raise ValueError("expected two sentiment scores")

    positive_score = int(float(fields[0]))
    negative_score = int(float(fields[1]))

    if not 1 <= positive_score <= 5:
        raise ValueError("positive score must be in [1, 5]")
    if not -5 <= negative_score <= -1:
        raise ValueError("negative score must be in [-5, -1]")

    negative_strength = abs(negative_score)
    net_score = positive_score - negative_strength

    if net_score >= 2:
        label = "Positive"
    elif net_score <= -2:
        label = "Negative"
    else:
        label = "Neutral"

    return positive_score, negative_score, net_score, label


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_file(input_path, output_dir, sep=FIELD_SEP):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = {label: output_dir / f"{label}.txt" for label in LABELS}
    index_paths = {
        label: output_dir / f"{label}-indices.txt" for label in LABELS
    }
    audit_path = output_dir / "sentiment_classification_audit.csv"
    manifest_path = output_dir / "sentiment_classification_manifest.json"

    data_files = {
        label: path.open("w", encoding="utf-8", newline="")
        for label, path in data_paths.items()
    }
    index_files = {
        label: path.open("w", encoding="utf-8", newline="")
        for label, path in index_paths.items()
    }

    counts = Counter()
    pair_counts = Counter()
    total = 0

    try:
        with input_path.open("r", encoding="utf-8", errors="strict") as source, \
                audit_path.open("w", encoding="utf-8-sig", newline="") as audit_file:
            writer = csv.writer(audit_file)
            writer.writerow([
                "line_number",
                "positive_score",
                "negative_score",
                "net_score",
                "label",
            ])

            for line_number, line in enumerate(source, start=1):
                total += 1
                raw_line = line.rstrip("\r\n")
                parts = raw_line.split(sep)
                if len(parts) <= SENTIMENT_INDEX:
                    raise ValueError(
                        f"Line {line_number}: missing eighth sentiment field"
                    )

                sentiment_pair = parts[SENTIMENT_INDEX].strip()
                try:
                    positive, negative, net, label = classify_sentiment(
                        sentiment_pair
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Line {line_number}: invalid sentiment pair "
                        f"{sentiment_pair!r} ({exc})"
                    ) from exc

                data_files[label].write(raw_line + "\n")
                index_files[label].write(f"{line_number}\n")
                writer.writerow([line_number, positive, negative, net, label])
                counts[label] += 1
                pair_counts[f"{positive} {negative}"] += 1
    finally:
        for handle in data_files.values():
            handle.close()
        for handle in index_files.values():
            handle.close()

    if sum(counts.values()) != total:
        raise RuntimeError("classified record count does not match input count")

    manifest = {
        "schema_version": 1,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "total_records": total,
        "line_numbering": "1-based; the first line of the input file is line 1",
        "sentiment_field": 8,
        "rule": {
            "net_score": "positive_score - abs(negative_score)",
            "Positive": "net_score >= 2",
            "Neutral": "-1 <= net_score <= 1",
            "Negative": "net_score <= -2",
        },
        "class_counts": {label: counts[label] for label in LABELS},
        "pair_counts": dict(sorted(pair_counts.items())),
        "files": {
            label: {
                "records": str(data_paths[label]),
                "indices": str(index_paths[label]),
            }
            for label in LABELS
        },
        "audit_file": str(audit_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Total: {total}")
    for label in LABELS:
        print(f"{label}: {counts[label]}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify records by their final two sentiment scores."
    )
    parser.add_argument("--input", default="./data/test.txt")
    parser.add_argument("--output-dir", default="./data/sentiment")
    parser.add_argument("--sep", default=FIELD_SEP)
    args = parser.parse_args()

    classify_file(args.input, args.output_dir, args.sep)


if __name__ == "__main__":
    main()
