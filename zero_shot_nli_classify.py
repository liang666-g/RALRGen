#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot NLI classifier for app-review evolution relevance.

Input format:
    Each record is one line with fields separated by literal "-[split]-".
    ONLY the 5th field (index 4) is used for classification.

Outputs:
    Evolution-related.txt
    Non-evolution-related.txt
    classification_audit.csv

Important:
    The two TXT outputs preserve the COMPLETE original input lines unchanged,
    so they remain aligned with other prediction/reference files.
"""

import argparse
import csv
from pathlib import Path

import torch
from transformers import pipeline

SEP = "-[split]-"

LABEL_EVO = (
    "describing a software update, version change, feature change, removal, "
    "addition, or behavior change across app versions"
)

LABEL_NON = (
    "describing an app experience or problem without referring to software "
    "evolution or version changes"
)

CANDIDATE_LABELS = [LABEL_EVO, LABEL_NON]
HYPOTHESIS_TEMPLATE = "This app review is about {}."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to test.txt")
    p.add_argument(
        "--model",
        default="./hf_cache/hub/models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0/snapshots/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2",
        help=(
            "Hugging Face model id or local model directory. "
            "Default: MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
        ),
    )
    p.add_argument("--output-dir", default=".", help="Directory for output files")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evo_path = out_dir / "Evolution-related.txt"
    non_path = out_dir / "Non-evolution-related.txt"
    audit_path = out_dir / "classification_audit.csv"

    # Preserve original line endings/content.
    with input_path.open("r", encoding="utf-8", newline="") as f:
        original_lines = f.readlines()

    reviews = []
    for line_no, raw_line in enumerate(original_lines, start=1):
        content = raw_line.rstrip("\r\n")
        fields = content.split(SEP)
        if len(fields) != 8:
            raise ValueError(
                f"Line {line_no}: expected 8 fields separated by {SEP!r}, "
                f"but found {len(fields)}."
            )
        # ONLY the 5th field is used for classification.
        reviews.append(fields[4].strip())

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=device,
    )

    predicted_labels = []
    predicted_scores = []

    for start in range(0, len(reviews), args.batch_size):
        batch = reviews[start : start + args.batch_size]

        results = classifier(
            batch,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            truncation=True,
            batch_size=args.batch_size,
        )

        # pipeline returns one dict per input for batched input.
        if isinstance(results, dict):
            results = [results]

        for result in results:
            best_label = result["labels"][0]
            best_score = float(result["scores"][0])
            predicted_labels.append(best_label)
            predicted_scores.append(best_score)

    if len(predicted_labels) != len(original_lines):
        raise RuntimeError(
            f"Prediction count mismatch: {len(predicted_labels)} predictions "
            f"for {len(original_lines)} input records."
        )

    evo_lines = []
    non_lines = []

    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["line_number", "review_text", "predicted_group", "confidence"]
        )

        for idx, (raw_line, review, label, score) in enumerate(
            zip(original_lines, reviews, predicted_labels, predicted_scores),
            start=1,
        ):
            if label == LABEL_EVO:
                group = "Evolution-related"
                evo_lines.append(raw_line)
            elif label == LABEL_NON:
                group = "Non-evolution-related"
                non_lines.append(raw_line)
            else:
                raise RuntimeError(f"Unexpected model label: {label!r}")

            writer.writerow([idx, review, group, f"{score:.8f}"])

    with evo_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(evo_lines)

    with non_path.open("w", encoding="utf-8", newline="") as f:
        f.writelines(non_lines)

    assert len(evo_lines) + len(non_lines) == len(original_lines)

    print(f"Total records: {len(original_lines)}")
    print(f"Evolution-related: {len(evo_lines)}")
    print(f"Non-evolution-related: {len(non_lines)}")
    print(f"Saved: {evo_path}")
    print(f"Saved: {non_path}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
