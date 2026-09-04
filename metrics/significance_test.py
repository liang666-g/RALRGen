"""比较两个回复生成模型的 BLEU 显著性与效应量。

推荐用法（逐样本、平滑 sentence BLEU）：

    python metrics/significance_test.py \
        --test-file data/test.txt \
        --model-a data/evaluation/model_a/predictions.txt \
        --model-b data/evaluation/model_b/predictions.txt \
        --model-a-name FFSGen \
        --model-b-name Baseline \
        --output data/evaluation/significance_ffsg_vs_baseline.json

若要尽量复现旧版 ttest_bleu.py 的“分块 corpus BLEU”做法，可增加：

    --unit block --block-size 1550

脚本不会保证得到显著结果。只有数据实际满足 p < alpha 且
|Cliff's delta| >= large_effect_threshold 时，报告才会标记为显著且大效应。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import nltk
import scipy
from nltk.translate.bleu_score import SmoothingFunction, modified_precision, sentence_bleu
from scipy import stats

try:
    # 以模块方式运行：python -m metrics.significance_test
    from .bleu import compute_bleu
except ImportError:
    # 以脚本方式运行：python metrics/significance_test.py
    from bleu import compute_bleu


DEFAULT_SEPARATOR = "-[split]-"
DEFAULT_INVALID_REPLIES = ("API调用失败",)
CLIFF_THRESHOLDS = {
    "negligible": 0.147,
    "small": 0.330,
    "medium": 0.474,
}


@dataclass(frozen=True)
class TestSample:
    source_line_no: int
    review: str
    reference: str


@dataclass(frozen=True)
class Prediction:
    source_line_no: int
    review: str
    reply: str


@dataclass(frozen=True)
class AlignedSample:
    test_line_no: int
    review: str
    reference: str
    model_a_reply: str
    model_b_reply: str


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\r\n") for line in f]


def load_test_samples(path: str, separator: str) -> Tuple[List[TestSample], int]:
    """读取与 tester._load_test_data() 相同口径的有效测试行。"""
    samples: List[TestSample] = []
    malformed = 0
    for line_no, line in enumerate(_read_lines(path), start=1):
        terms = line.split(separator)
        if len(terms) < 8:
            malformed += 1
            continue
        samples.append(
            TestSample(
                source_line_no=line_no,
                review=terms[4],
                reference=terms[5],
            )
        )
    if not samples:
        raise ValueError(f"测试文件没有可用数据：{path}")
    return samples, malformed


def load_predictions(path: str, separator: str) -> Tuple[List[Prediction], List[int]]:
    """读取 review-[split]-reply，并跳过不能精确分成两部分的异常行。"""
    predictions: List[Prediction] = []
    malformed_line_numbers: List[int] = []
    for line_no, line in enumerate(_read_lines(path), start=1):
        terms = line.split(separator)
        if len(terms) != 2:
            malformed_line_numbers.append(line_no)
            continue
        predictions.append(
            Prediction(
                source_line_no=line_no,
                review=terms[0],
                reply=terms[1],
            )
        )
    return predictions, malformed_line_numbers


def _normalize_invalid_reply(text: str) -> str:
    return "".join(str(text or "").strip().split())


def _invalid_reply_reason(reply: str, invalid_reply_set: set[str]) -> str | None:
    stripped = str(reply or "").strip()
    if not stripped:
        return "empty_reply"
    if stripped.startswith("{"):
        return "json_reply_startswith_left_brace"
    if _normalize_invalid_reply(stripped) in invalid_reply_set:
        return "invalid_placeholder_reply"
    return None


def align_common_valid_samples(
    test_samples: Sequence[TestSample],
    model_a: Sequence[Prediction],
    model_b: Sequence[Prediction],
    invalid_replies: Iterable[str],
) -> Tuple[List[AlignedSample], Dict[str, object]]:
    """严格按行对齐，并对两个模型使用共同有效样本掩码。"""
    expected = len(test_samples)
    if len(model_a) != expected or len(model_b) != expected:
        raise ValueError(
            "删除格式异常的预测行后，预测数量必须与测试样本数量一致，"
            "否则无法保证成对检验的行级对齐。\n"
            f"test={expected}, model_a={len(model_a)}, model_b={len(model_b)}"
        )

    mismatches: List[Dict[str, object]] = []
    for index, (test, pred_a, pred_b) in enumerate(
        zip(test_samples, model_a, model_b), start=1
    ):
        if pred_a.review != test.review or pred_b.review != test.review:
            mismatches.append(
                {
                    "aligned_index": index,
                    "test_line_no": test.source_line_no,
                    "model_a_line_no": pred_a.source_line_no,
                    "model_b_line_no": pred_b.source_line_no,
                    "test_review": test.review[:200],
                    "model_a_review": pred_a.review[:200],
                    "model_b_review": pred_b.review[:200],
                }
            )
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError(
            "预测文件与测试集的评论列没有按行对齐。前几个错位样本：\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )

    invalid_reply_set = {
        _normalize_invalid_reply(x)
        for x in invalid_replies
        if str(x or "").strip()
    }
    dropped_reasons: Counter[str] = Counter()
    aligned: List[AlignedSample] = []

    for test, pred_a, pred_b in zip(test_samples, model_a, model_b):
        reason_a = _invalid_reply_reason(pred_a.reply, invalid_reply_set)
        reason_b = _invalid_reply_reason(pred_b.reply, invalid_reply_set)
        if reason_a or reason_b:
            if reason_a:
                dropped_reasons[f"model_a:{reason_a}"] += 1
            if reason_b:
                dropped_reasons[f"model_b:{reason_b}"] += 1
            dropped_reasons["common_rows_dropped"] += 1
            continue
        aligned.append(
            AlignedSample(
                test_line_no=test.source_line_no,
                review=test.review,
                reference=test.reference,
                model_a_reply=pred_a.reply,
                model_b_reply=pred_b.reply,
            )
        )

    if not aligned:
        raise ValueError("两个模型没有共同有效的对齐样本。")

    return aligned, {
        "test_samples_before_common_filter": expected,
        "common_valid_samples": len(aligned),
        "common_rows_dropped": int(dropped_reasons.get("common_rows_dropped", 0)),
        "drop_reason_counts": dict(sorted(dropped_reasons.items())),
    }


def _bleu_for_corpus(
    samples: Sequence[AlignedSample],
    prediction_field: str,
    *,
    smooth: bool,
) -> Tuple[float, List[float]]:
    references = [[sample.reference.strip().split()] for sample in samples]
    candidates = [str(getattr(sample, prediction_field)).strip().split() for sample in samples]

    if len(samples) == 1 and smooth:
        # 项目自带 compute_bleu 的 add-one smooth 会让“不存在的高阶 n-gram”
        # 精度变成 1。逐样本检验改用 NLTK 标准 sentence BLEU method1。
        refs = references[0]
        candidate = candidates[0]
        bleu = sentence_bleu(
            refs,
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
        precisions = [float(modified_precision(refs, candidate, order)) for order in range(1, 5)]
        return float(bleu), precisions

    bleu, precisions, _, _, _, _ = compute_bleu(
        references,
        candidates,
        max_order=4,
        smooth=smooth,
    )
    return float(bleu), [float(x) for x in precisions]


def build_metric_scores(
    samples: Sequence[AlignedSample],
    *,
    unit: str,
    block_size: int,
    include_precisions: bool,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    """生成 Wilcoxon 所需的成对分数序列。"""
    if unit == "sample":
        groups = [[sample] for sample in samples]
        smooth = True
    elif unit == "block":
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0。")
        groups = [list(samples[i : i + block_size]) for i in range(0, len(samples), block_size)]
        smooth = False
        if len(groups) < 2:
            raise ValueError(
                "分块后至少需要两个统计单元；请减小 --block-size 或改用 --unit sample。"
            )
    else:
        raise ValueError(f"未知统计单位：{unit}")

    names = ["BLEU-4"]
    if include_precisions:
        names.extend(["1-gram_precision", "2-gram_precision", "3-gram_precision", "4-gram_precision"])

    values_a: Dict[str, List[float]] = {name: [] for name in names}
    values_b: Dict[str, List[float]] = {name: [] for name in names}

    for group in groups:
        bleu_a, precisions_a = _bleu_for_corpus(group, "model_a_reply", smooth=smooth)
        bleu_b, precisions_b = _bleu_for_corpus(group, "model_b_reply", smooth=smooth)
        values_a["BLEU-4"].append(bleu_a)
        values_b["BLEU-4"].append(bleu_b)

        if include_precisions:
            for order in range(1, 5):
                key = f"{order}-gram_precision"
                values_a[key].append(precisions_a[order - 1])
                values_b[key].append(precisions_b[order - 1])

    pairs = {
        name: (
            np.asarray(values_a[name], dtype=np.float64),
            np.asarray(values_b[name], dtype=np.float64),
        )
        for name in names
    }
    return pairs, {
        "unit": unit,
        "statistical_units": len(groups),
        "block_size": block_size if unit == "block" else None,
        "block_lengths": [len(group) for group in groups] if unit == "block" else None,
        "bleu_smoothing": smooth,
        "bleu_implementation": (
            "NLTK sentence_bleu + SmoothingFunction.method1"
            if unit == "sample"
            else "project metrics.bleu.compute_bleu corpus BLEU without smoothing"
        ),
    }


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """普通（非配对）Cliff's Delta；正值表示模型 A 的得分倾向更高。"""
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Cliff's Delta 需要两个非空分数序列。")
    u = float(stats.mannwhitneyu(a, b, alternative="two-sided", method="auto").statistic)
    return 2.0 * u / float(len(a) * len(b)) - 1.0


def cliff_magnitude(delta: float) -> str:
    value = abs(float(delta))
    if value < CLIFF_THRESHOLDS["negligible"]:
        return "negligible"
    if value < CLIFF_THRESHOLDS["small"]:
        return "small"
    if value < CLIFF_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def paired_rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """与 Wilcoxon 配对设计一致的补充效应量。"""
    differences = np.asarray(a - b, dtype=np.float64)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def _safe_sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_metric(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternative: str,
    alpha: float,
    large_effect_threshold: float,
) -> Dict[str, object]:
    if a.shape != b.shape:
        raise ValueError(f"{name} 的成对分数数量不一致：{a.shape} vs {b.shape}")
    if len(a) < 2:
        raise ValueError(f"{name} 至少需要两个成对统计单元。")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{name} 含有 NaN 或无穷值。")

    differences = a - b
    nonzero = int(np.count_nonzero(differences))
    if nonzero == 0:
        wilcoxon_statistic = 0.0
        p_value = 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            a,
            b,
            zero_method="wilcox",
            alternative=alternative,
            method="auto",
        )
        wilcoxon_statistic = float(wilcoxon_result.statistic)
        p_value = float(wilcoxon_result.pvalue)

    delta = float(cliffs_delta(a, b))
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_difference = mean_a - mean_b

    return {
        "metric": name,
        "n_pairs": int(len(a)),
        "model_a": {
            "mean": mean_a,
            "median": float(np.median(a)),
            "sample_std": _safe_sample_std(a),
        },
        "model_b": {
            "mean": mean_b,
            "median": float(np.median(b)),
            "sample_std": _safe_sample_std(b),
        },
        "paired_difference": {
            "mean": mean_difference,
            "median": float(np.median(differences)),
            "wins": int(np.count_nonzero(differences > 0)),
            "ties": int(np.count_nonzero(differences == 0)),
            "losses": int(np.count_nonzero(differences < 0)),
            "nonzero_pairs_used_by_wilcoxon": nonzero,
        },
        "wilcoxon": {
            "statistic": wilcoxon_statistic,
            "p_value_raw": p_value,
            "p_value_holm": None,
            "alternative": alternative,
            "zero_method": "wilcox",
            "significant_raw": bool(p_value < alpha),
            "significant_holm": None,
        },
        "effect_sizes": {
            "cliffs_delta": delta,
            "cliffs_delta_absolute": abs(delta),
            "cliffs_delta_magnitude": cliff_magnitude(delta),
            "paired_rank_biserial": float(paired_rank_biserial(a, b)),
            "large_effect": bool(abs(delta) >= large_effect_threshold),
        },
        "model_a_mean_higher": bool(mean_difference > 0),
        "significant_large_improvement_raw": bool(
            p_value < alpha and abs(delta) >= large_effect_threshold and mean_difference > 0
        ),
        "significant_large_improvement_holm": None,
    }


def apply_holm_correction(results: List[Dict[str, object]], alpha: float) -> None:
    """原地写入 Holm 校正后的 p 值和最终判定。"""
    indexed = sorted(
        enumerate(results),
        key=lambda item: float(item[1]["wilcoxon"]["p_value_raw"]),  # type: ignore[index]
    )
    total = len(indexed)
    running_max = 0.0
    adjusted_by_index: Dict[int, float] = {}
    for rank, (original_index, result) in enumerate(indexed):
        raw = float(result["wilcoxon"]["p_value_raw"])  # type: ignore[index]
        adjusted = min(1.0, (total - rank) * raw)
        running_max = max(running_max, adjusted)
        adjusted_by_index[original_index] = running_max

    for index, result in enumerate(results):
        adjusted = adjusted_by_index[index]
        wilcoxon = result["wilcoxon"]  # type: ignore[assignment]
        effects = result["effect_sizes"]  # type: ignore[assignment]
        wilcoxon["p_value_holm"] = adjusted
        wilcoxon["significant_holm"] = bool(adjusted < alpha)
        result["significant_large_improvement_holm"] = bool(
            adjusted < alpha
            and bool(effects["large_effect"])
            and bool(result["model_a_mean_higher"])
        )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_model_name(path: str) -> str:
    file_path = Path(path)
    return file_path.parent.name or file_path.stem


def _write_scores_csv(
    path: str,
    metric_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_a_name: str,
    model_b_name: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "unit_index", model_a_name, model_b_name, "difference_a_minus_b"])
        for metric, (a, b) in metric_pairs.items():
            for index, (score_a, score_b) in enumerate(zip(a, b), start=1):
                writer.writerow([metric, index, score_a, score_b, score_a - score_b])


def _print_report(report: Dict[str, object]) -> None:
    alignment = report["alignment"]  # type: ignore[assignment]
    settings = report["settings"]  # type: ignore[assignment]
    print("=" * 92)
    print("Wilcoxon + Cliff's Delta 显著性检验")
    print("=" * 92)
    print(
        f"共同有效样本: {alignment['common_valid_samples']} | "
        f"统计单位: {settings['unit']} | 单元数: {settings['statistical_units']} | "
        f"alpha: {settings['alpha']}"
    )
    print(
        f"模型 A: {report['models']['model_a']['name']}\n"  # type: ignore[index]
        f"模型 B: {report['models']['model_b']['name']}"  # type: ignore[index]
    )
    print("-" * 92)
    header = (
        f"{'Metric':<21} {'Mean A':>10} {'Mean B':>10} {'A-B':>10} "
        f"{'p(raw)':>11} {'p(Holm)':>11} {'Cliff d':>10} {'Effect':>11} {'Pass':>7}"
    )
    print(header)
    print("-" * 92)
    for result in report["results"]:  # type: ignore[assignment]
        passed = "YES" if result["significant_large_improvement_holm"] else "NO"
        print(
            f"{result['metric']:<21} "
            f"{result['model_a']['mean']:>10.6f} "
            f"{result['model_b']['mean']:>10.6f} "
            f"{result['paired_difference']['mean']:>10.6f} "
            f"{result['wilcoxon']['p_value_raw']:>11.3g} "
            f"{result['wilcoxon']['p_value_holm']:>11.3g} "
            f"{result['effect_sizes']['cliffs_delta']:>10.6f} "
            f"{result['effect_sizes']['cliffs_delta_magnitude']:>11} "
            f"{passed:>7}"
        )
    print("=" * 92)
    print(
        "Pass=YES 表示：Holm 校正 p < alpha、|Cliff d| 达到大效应阈值，"
        "且模型 A 的平均分高于模型 B。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在严格对齐的两个模型输出上执行 Wilcoxon 与 Cliff's Delta。"
    )
    parser.add_argument("--test-file", required=True, help="8 字段 test.txt")
    parser.add_argument("--model-a", required=True, help="模型 A 的 predictions.txt")
    parser.add_argument("--model-b", required=True, help="模型 B 的 predictions.txt")
    parser.add_argument("--model-a-name", default="", help="报告中的模型 A 名称")
    parser.add_argument("--model-b-name", default="", help="报告中的模型 B 名称")
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR)
    parser.add_argument(
        "--unit",
        choices=("sample", "block"),
        default="sample",
        help="sample=平滑句级 BLEU（推荐）；block=分块 corpus BLEU（复现旧脚本）",
    )
    parser.add_argument("--block-size", type=int, default=1550)
    parser.add_argument(
        "--include-ngram-precisions",
        action="store_true",
        help="同时检验 1~4 gram precision，并对全部 p 值做 Holm 校正",
    )
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help="Wilcoxon 备择假设；greater 表示预先假设模型 A 更高",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--large-effect-threshold", type=float, default=0.474)
    parser.add_argument(
        "--invalid-reply",
        action="append",
        default=None,
        help="需要剔除的失败占位回复；可重复传入，默认包含 API调用失败",
    )
    parser.add_argument("--output", default="", help="可选：JSON 报告输出路径")
    parser.add_argument("--scores-csv", default="", help="可选：保存每个统计单元的成对分数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha 必须位于 (0, 1)。")
    if not (0.0 <= args.large_effect_threshold <= 1.0):
        raise ValueError("--large-effect-threshold 必须位于 [0, 1]。")

    for path in (args.test_file, args.model_a, args.model_b):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    model_a_name = args.model_a_name.strip() or _default_model_name(args.model_a)
    model_b_name = args.model_b_name.strip() or _default_model_name(args.model_b)
    invalid_replies = list(DEFAULT_INVALID_REPLIES)
    if args.invalid_reply:
        invalid_replies.extend(args.invalid_reply)

    test_samples, malformed_test = load_test_samples(args.test_file, args.separator)
    model_a, malformed_a = load_predictions(args.model_a, args.separator)
    model_b, malformed_b = load_predictions(args.model_b, args.separator)
    aligned, alignment = align_common_valid_samples(
        test_samples,
        model_a,
        model_b,
        invalid_replies,
    )
    alignment.update(
        {
            "malformed_test_lines_skipped": malformed_test,
            "malformed_model_a_lines_skipped": len(malformed_a),
            "malformed_model_b_lines_skipped": len(malformed_b),
            "malformed_model_a_line_examples": malformed_a[:10],
            "malformed_model_b_line_examples": malformed_b[:10],
        }
    )

    metric_pairs, unit_info = build_metric_scores(
        aligned,
        unit=args.unit,
        block_size=args.block_size,
        include_precisions=args.include_ngram_precisions,
    )
    results = [
        evaluate_metric(
            name,
            scores_a,
            scores_b,
            alternative=args.alternative,
            alpha=args.alpha,
            large_effect_threshold=args.large_effect_threshold,
        )
        for name, (scores_a, scores_b) in metric_pairs.items()
    ]
    apply_holm_correction(results, args.alpha)

    report: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "significance_test": "Wilcoxon signed-rank test",
            "effect_size_primary": "Cliff's Delta (ordinary/unpaired definition)",
            "effect_size_supplementary": "paired rank-biserial correlation",
            "cliffs_delta_thresholds": {
                "negligible": "|d| < 0.147",
                "small": "0.147 <= |d| < 0.330",
                "medium": "0.330 <= |d| < 0.474",
                "large": "|d| >= 0.474",
            },
            "multiple_testing_correction": "Holm",
        },
        "settings": {
            **unit_info,
            "alpha": args.alpha,
            "alternative": args.alternative,
            "large_effect_threshold": args.large_effect_threshold,
            "include_ngram_precisions": bool(args.include_ngram_precisions),
            "separator": args.separator,
            "invalid_replies": list(invalid_replies),
        },
        "files": {
            "test": {"path": os.path.abspath(args.test_file), "sha256": _sha256(args.test_file)},
            "model_a": {"path": os.path.abspath(args.model_a), "sha256": _sha256(args.model_a)},
            "model_b": {"path": os.path.abspath(args.model_b), "sha256": _sha256(args.model_b)},
        },
        "models": {
            "model_a": {"name": model_a_name},
            "model_b": {"name": model_b_name},
        },
        "alignment": alignment,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "nltk": nltk.__version__,
            "scipy": scipy.__version__,
        },
        "results": results,
    }

    _print_report(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"JSON 报告: {output.resolve()}")

    if args.scores_csv:
        _write_scores_csv(args.scores_csv, metric_pairs, model_a_name, model_b_name)
        print(f"成对分数: {Path(args.scores_csv).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
