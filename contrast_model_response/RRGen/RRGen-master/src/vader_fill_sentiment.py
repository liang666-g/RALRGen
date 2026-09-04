# -*- coding: utf-8 -*-
"""
Use VADER to fill the 8th sentiment field for RRGen-style app review data.

Input format:
app-[split]-rating-[split]-field2-[split]-field3-[split]-review-[split]-reply-[split]-field6-[split]-content

Output format:
app-[split]-rating-[split]-field2-[split]-field3-[split]-review-[split]-reply-[split]-field6-[split]-positive_score negative_score

The output sentiment field is compatible with the existing RRGen code:
    positive_score: integer in [1, 5]
    negative_score: integer in [-5, -1]
"""

import argparse
import os
import re
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as exc:
    raise ImportError(
        "Missing dependency: vaderSentiment. Install it with:\n"
        "    pip install vaderSentiment\n"
    ) from exc


FIELD_SEP = "-[split]-"
REVIEW_INDEX = 4
SENTIMENT_INDEX = 7


def clean_review_text(text):
    """
    Keep the text close to the original review, but remove placeholders that
    can confuse a generic sentiment analyzer.
    """
    if text is None:
        return ""

    text = text.replace("<URL>", " ")
    text = text.replace("<url>", " ")
    text = text.replace("<EMAIL>", " ")
    text = text.replace("<email>", " ")
    text = text.replace("<USER>", " ")
    text = text.replace("<user>", " ")
    text = text.replace("<APP>", " ")
    text = text.replace("<app>", " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def vader_to_sentistrength_like(text, analyzer):
    """
    Convert VADER output to a SentiStrength-like pair.

    SentiStrength original style:
        positive_score in [1, 5]
        negative_score in [-5, -1]

    VADER gives:
        pos, neu, neg in [0, 1]
        compound in [-1, 1]

    This function maps VADER intensity to the same two-integer format expected
    by the existing RRGen data loader.
    """
    cleaned = clean_review_text(text)

    if not cleaned or cleaned.lower() == "content":
        return "1 -1"

    scores = analyzer.polarity_scores(cleaned)

    compound = scores["compound"]
    pos_ratio = scores["pos"]
    neg_ratio = scores["neg"]

    # Use both local positive/negative ratio and global compound score.
    # This keeps mixed reviews from being collapsed too aggressively.
    pos_intensity = max(pos_ratio, max(compound, 0.0))
    neg_intensity = max(neg_ratio, max(-compound, 0.0))

    pos_score = int(round(1 + 4 * clamp(pos_intensity, 0.0, 1.0)))
    neg_abs_score = int(round(1 + 4 * clamp(neg_intensity, 0.0, 1.0)))

    pos_score = clamp(pos_score, 1, 5)
    neg_abs_score = clamp(neg_abs_score, 1, 5)

    neg_score = -neg_abs_score

    return "{} {}".format(pos_score, neg_score)


def selected_sentiment_class(sentiment_pair):
    """
    Replicate the existing RRGen loader's decision rule.

    Existing logic:
        if negative * 1.5 + positive < 0:
            choose negative
        else:
            choose positive
    """
    try:
        positive, negative = sentiment_pair.split()
        positive = int(positive)
        negative = int(negative)

        if negative * 1.5 + positive < 0:
            return str(negative)
        return str(positive)
    except Exception:
        return "INVALID"


def process_file(input_path, output_path, sep=FIELD_SEP, replace_only_content=False):
    analyzer = SentimentIntensityAnalyzer()

    total = 0
    written = 0
    skipped = 0
    selected_counter = Counter()
    pair_counter = Counter()

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fr, \
            open(output_path, "w", encoding="utf-8") as fw:

        for line in tqdm(fr, desc="Processing {}".format(input_path)):
            total += 1
            raw_line = line.rstrip("\n")
            parts = raw_line.split(sep)

            # Need at least app, rating, review, reply positions.
            if len(parts) <= REVIEW_INDEX:
                skipped += 1
                continue

            # Pad to 8 fields if the line is shorter.
            while len(parts) <= SENTIMENT_INDEX:
                parts.append("content")

            old_sentiment = parts[SENTIMENT_INDEX].strip()

            if replace_only_content and old_sentiment.lower() != "content":
                sentiment_pair = old_sentiment
            else:
                review_text = parts[REVIEW_INDEX]
                sentiment_pair = vader_to_sentistrength_like(review_text, analyzer)
                parts[SENTIMENT_INDEX] = sentiment_pair

            pair_counter[sentiment_pair] += 1
            selected_counter[selected_sentiment_class(sentiment_pair)] += 1

            fw.write(sep.join(parts) + "\n")
            written += 1

    print("=" * 80)
    print("Input file :", input_path)
    print("Output file:", output_path)
    print("Total lines:", total)
    print("Written    :", written)
    print("Skipped    :", skipped)
    print("- Sentiment pair distribution:")
    for k, v in pair_counter.most_common():
        print("  {} -> {}".format(k, v))
    print("- Final sentiment class distribution used by RRGen loader:")
    for k, v in selected_counter.most_common():
        print("  {} -> {}".format(k, v))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Fill the 8th sentiment field using VADER for RRGen data."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input data file, e.g. ./src/data/train.txt"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output data file, e.g. ./src/data/raw_train.txt"
    )
    parser.add_argument(
        "--sep",
        default=FIELD_SEP,
        help="Field separator. Default: -[split]-"
    )
    parser.add_argument(
        "--replace-only-content",
        action="store_true",
        help="Only replace the 8th field when it is exactly 'content'."
    )

    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        sep=args.sep,
        replace_only_content=args.replace_only_content
    )


if __name__ == "__main__":
    main()