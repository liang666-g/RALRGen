from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Any


def _read_text_lines(file_path: str) -> List[str]:
    """按行读取文本，去掉每行末尾换行符。"""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def _write_text_lines(file_path: str, lines: List[str]) -> None:
    """按行写出文本。"""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip("\n") + "\n")


def load_valid_test_lines(test_file: str) -> List[str]:
    """
    读取 test 文件中真正参与生成/评测的有效行。

    与 tester._load_test_data() 保持一致：
    - line.split("-[split]-") 后字段数小于 8 的行会被跳过。
    """
    valid_lines = []
    with open(test_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.rstrip("\n")
            terms = raw.split("-[split]-")
            if len(terms) < 8:
                continue
            valid_lines.append(raw)
    return valid_lines


def clean_pred_and_test_for_eval(
    pred_file: str,
    test_file: str,
    out_dir: str | None = None,
    sep: str = "-[split]-",
) -> Dict[str, Any]:
    """
    清洗预测文件，并同步生成对齐后的 test 文件。

    清洗步骤：
    1. 解析预测行中的 source/reply。只切分第一个 sep，保留 reply 中
       可能再次出现的 sep。
    2. 检查可解析预测行数是否等于 test 有效行数。
       若不一致，停止评测，避免错位计算指标。
    3. 保留所有已解析的回复，包括空回复、JSON 回复和 API 失败文本。

    返回：
        一个 dict，包含清洗后的文件路径、删除记录和统计信息。
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pred_file))
    os.makedirs(out_dir, exist_ok=True)

    raw_pred_lines = _read_text_lines(pred_file)
    valid_test_lines = load_valid_test_lines(test_file)

    # 解析预测行，不根据 reply 内容删除样本。
    # 只切分第一个分隔符，允许 reply 正文中再次出现 sep。
    split_clean_items = []
    malformed_rows = []

    for raw_line_no, line in enumerate(raw_pred_lines, start=1):
        parts = line.split(sep, 1)
        if len(parts) != 2:
            malformed_rows.append(
                {
                    "raw_prediction_line_no": raw_line_no,
                    "reason": "not_split_into_two_parts_by_separator",
                    "line": line,
                }
            )
            continue

        split_clean_items.append(
            {
                "raw_prediction_line_no": raw_line_no,
                "line": line,
            }
        )

    split_clean_pred_lines = [item["line"] for item in split_clean_items]
    split_clean_pred_file = os.path.join(out_dir, "predictions.clean_split.txt")
    _write_text_lines(split_clean_pred_file, split_clean_pred_lines)

    # 无法解析的行不再静默删除；即使删除后数量碰巧相等，也必须停止。
    # 这样可以保证没有任何预测回复被悄悄排除在评测之外。
    if malformed_rows or len(split_clean_pred_lines) != len(valid_test_lines):
        mismatch_report = {
            "status": "line_count_mismatch_after_split_clean",
            "pred_file": pred_file,
            "test_file": test_file,
            "raw_prediction_lines": len(raw_pred_lines),
            "prediction_lines_after_split_clean": len(split_clean_pred_lines),
            "valid_test_lines": len(valid_test_lines),
            "malformed_rows_not_used": len(malformed_rows),
            "malformed_rows": malformed_rows,
            "split_clean_pred_file": split_clean_pred_file,
        }

        report_file = os.path.join(out_dir, "clean_report_mismatch.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, ensure_ascii=False, indent=4)

        raise ValueError(
            "评测失败：预测文件存在无法解析的行，或预测数量与 test 有效行数不一致。"
            "为避免删除样本或错位评测，已停止。"
            f"\n原始预测行数: {len(raw_pred_lines)}"
            f"\n可解析预测行数: {len(split_clean_pred_lines)}"
            f"\ntest 有效行数: {len(valid_test_lines)}"
            f"\n详情报告: {report_file}"
        )

    # 数量一致后原样保留全部可解析预测和对应 test 行。
    # 空回复、JSON 回复和 API 失败文本都进入后续指标计算。
    final_pred_lines = split_clean_pred_lines
    final_test_lines = valid_test_lines
    deleted_aligned_rows = []

    final_pred_file = os.path.join(out_dir, "predictions.clean_valid.txt")
    final_test_file = os.path.join(out_dir, "test.clean_valid.txt")
    deleted_rows_file = os.path.join(out_dir, "deleted_rows.json")

    _write_text_lines(final_pred_file, final_pred_lines)
    _write_text_lines(final_test_file, final_test_lines)

    result = {
        "status": "ok",
        "pred_file": pred_file,
        "test_file": test_file,
        "raw_prediction_lines": len(raw_pred_lines),
        "valid_test_lines": len(valid_test_lines),
        "malformed_rows_not_used": len(malformed_rows),
        "prediction_lines_after_split_clean": len(split_clean_pred_lines),
        "content_filtering_enabled": False,
        "invalid_aligned_rows_removed_second": len(deleted_aligned_rows),
        "invalid_placeholder_reply_texts": [],
        "invalid_placeholder_rows_removed_second": 0,
        "final_prediction_lines": len(final_pred_lines),
        "final_test_lines": len(final_test_lines),
        "split_clean_pred_file": split_clean_pred_file,
        "final_pred_file": final_pred_file,
        "final_test_file": final_test_file,
        "deleted_rows_file": deleted_rows_file,
        "malformed_rows": malformed_rows,
        "deleted_aligned_rows": deleted_aligned_rows,
        "temp_files": [
            split_clean_pred_file,
            final_pred_file,
            final_test_file,
            deleted_rows_file,
        ],
    }

    with open(deleted_rows_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    k: v
                    for k, v in result.items()
                    if k not in {"malformed_rows", "deleted_aligned_rows"}
                },
                "malformed_rows_not_used": malformed_rows,
                "deleted_aligned_rows_removed_second": deleted_aligned_rows,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("============== 预测文件清洗完成 ==============")
    print(f"原始预测文件行数: {len(raw_pred_lines)}")
    print(f"无法解析且未纳入评测的预测行数: {len(malformed_rows)}")
    print(f"可解析预测文件行数: {len(split_clean_pred_lines)}")
    print(f"test 有效行数: {len(valid_test_lines)}")
    print("回复内容过滤: 已关闭，所有可解析回复均保留")
    print(f"最终预测文件行数: {len(final_pred_lines)}")
    print(f"最终 test 文件行数: {len(final_test_lines)}")

    return result


def load_clean_predictions(clean_pred_file: str, sep: str = "-[split]-") -> List[str]:
    """
    从清洗后的预测文件中读取 reply 列表。

    clean_pred_file 理论上已经保证每行都能被 sep 精确切成两部分。
    """
    out_texts = []
    with open(clean_pred_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(sep, 1)
            if len(parts) == 2:
                out_texts.append(parts[1])
            else:
                out_texts.append("")
    return out_texts


def remove_temp_files(file_paths: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    删除中间文件。

    返回：
        removed: 成功删除的文件路径
        failed: 删除失败的文件及原因
    """
    removed = []
    failed = []

    seen = set()
    for fp in file_paths:
        if not fp:
            continue

        abs_fp = os.path.abspath(fp)
        if abs_fp in seen:
            continue
        seen.add(abs_fp)

        try:
            if os.path.exists(abs_fp):
                os.remove(abs_fp)
                removed.append(abs_fp)
        except Exception as e:
            failed.append(
                {
                    "file": abs_fp,
                    "error": str(e),
                }
            )

    return removed, failed


def write_metrics_and_cleanup(
    metrics_file: str,
    metrics_data: Dict[str, Any],
    clean_result: Dict[str, Any],
    remove_intermediate_files: bool = True,
) -> None:
    """
    写出 metrics.json，并按需删除清洗中间文件。

    这个函数会把清洗记录合并进 metrics.json：
    - cleaning_report：清洗统计、删除行号、删除原因
    - temp_file_cleanup：中间文件删除情况
    """
    metrics_data = dict(metrics_data)

    cleaning_report = {
        "summary": {
            k: v
            for k, v in clean_result.items()
            if k not in {"malformed_rows", "deleted_aligned_rows"}
        },
        "malformed_rows_removed_first": clean_result.get("malformed_rows", []),
        "deleted_aligned_rows_removed_second": clean_result.get("deleted_aligned_rows", []),
    }

    cleanup_report = {
        "removed": [],
        "failed": [],
    }

    if remove_intermediate_files:
        removed, failed = remove_temp_files(clean_result.get("temp_files", []))
        cleanup_report = {
            "removed": removed,
            "failed": failed,
        }
        print(f"已删除评测清洗中间文件 {len(removed)} 个。")
        if failed:
            print(f"[warn] 有 {len(failed)} 个中间文件删除失败，详情已写入 metrics.json。")

    metrics_data["cleaning_report"] = cleaning_report
    metrics_data["temp_file_cleanup"] = cleanup_report

    os.makedirs(os.path.dirname(os.path.abspath(metrics_file)), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
