from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Any, Optional, Iterable


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


def _normalize_reply_for_invalid_check(text: str) -> str:
    """
    归一化 reply，用于判断是否是需要从评测中剔除的占位错误文本。

    这里只做很轻量的归一化：
    - 去掉首尾空白
    - 去掉中间所有空白字符
    这样可以兼容 "API调用失败"、"API 调用失败"、"API调用失败\n" 等形式。
    """
    return "".join(str(text or "").strip().split())


def _build_invalid_reply_set(invalid_reply_texts: Optional[Iterable[str]]) -> set[str]:
    """构造需要剔除的 reply 文本集合。"""
    if invalid_reply_texts is None:
        invalid_reply_texts = ("API调用失败",)
    return {
        _normalize_reply_for_invalid_check(x)
        for x in invalid_reply_texts
        if str(x or "").strip()
    }


def _is_invalid_placeholder_reply(reply: str, invalid_reply_set: set[str]) -> bool:
    """判断 reply 是否命中需要同步删除的占位错误文本。"""
    normalized = _normalize_reply_for_invalid_check(reply)
    return normalized in invalid_reply_set


def clean_pred_and_test_for_eval(
    pred_file: str,
    test_file: str,
    out_dir: str | None = None,
    sep: str = "-[split]-",
    invalid_reply_texts: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    清洗预测文件，并同步生成对齐后的 test 文件。

    清洗步骤：
    1. 删除不能被 sep 精确划分成两部分的预测行。
       这些行通常是模型返回多行 JSON 后产生的残留行，不对应 test 中的真实样本。
    2. 检查第一步清洗后的预测行数是否等于 test 有效行数。
       若不一致，停止评测，避免错位计算指标。
    3. 在已经对齐的基础上，删除 reply 为空或 reply 以 "{" 开头的行。
       这些行对应真实样本，所以必须同步删除 test 文件中相同行号的数据。

    返回：
        一个 dict，包含清洗后的文件路径、删除记录和统计信息。
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pred_file))
    os.makedirs(out_dir, exist_ok=True)

    invalid_reply_set = _build_invalid_reply_set(invalid_reply_texts)

    raw_pred_lines = _read_text_lines(pred_file)
    valid_test_lines = load_valid_test_lines(test_file)

    # 第一步：只删除不能被 sep 精确划分成两部分的预测行
    split_clean_items = []
    malformed_rows = []

    for raw_line_no, line in enumerate(raw_pred_lines, start=1):
        parts = line.split(sep)
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

    # 第二步：第一步清洗后必须与 test 有效行数一致
    if len(split_clean_pred_lines) != len(valid_test_lines):
        mismatch_report = {
            "status": "line_count_mismatch_after_split_clean",
            "pred_file": pred_file,
            "test_file": test_file,
            "raw_prediction_lines": len(raw_pred_lines),
            "prediction_lines_after_split_clean": len(split_clean_pred_lines),
            "valid_test_lines": len(valid_test_lines),
            "malformed_rows_removed_first": len(malformed_rows),
            "malformed_rows": malformed_rows,
            "split_clean_pred_file": split_clean_pred_file,
        }

        report_file = os.path.join(out_dir, "clean_report_mismatch.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(mismatch_report, f, ensure_ascii=False, indent=4)

        raise ValueError(
            "清洗失败：删除不能被 '-[split]-' 精确划分成两部分的预测行后，"
            "预测文件行数仍然与 test 有效行数不一致，不能继续评测。"
            f"\n原始预测行数: {len(raw_pred_lines)}"
            f"\n第一步清洗后预测行数: {len(split_clean_pred_lines)}"
            f"\ntest 有效行数: {len(valid_test_lines)}"
            f"\n详情报告: {report_file}"
        )

    # 第三步：在已对齐基础上，同步删除无效 reply 行
    final_pred_lines = []
    final_test_lines = []
    deleted_aligned_rows = []

    for aligned_line_no, (pred_item, test_line) in enumerate(
        zip(split_clean_items, valid_test_lines),
        start=1,
    ):
        pred_line = pred_item["line"]
        src, reply = pred_line.split(sep)
        reply_stripped = reply.strip()

        delete_reason = None
        if not reply_stripped:
            delete_reason = "empty_reply"
        elif reply_stripped.startswith("{"):
            delete_reason = "json_reply_startswith_left_brace"
        elif _is_invalid_placeholder_reply(reply_stripped, invalid_reply_set):
            delete_reason = "invalid_placeholder_reply"

        if delete_reason:
            deleted_aligned_rows.append(
                {
                    "aligned_line_no": aligned_line_no,
                    "raw_prediction_line_no": pred_item["raw_prediction_line_no"],
                    "reason": delete_reason,
                    "prediction_line": pred_line,
                    "test_line": test_line,
                }
            )
            continue

        final_pred_lines.append(pred_line)
        final_test_lines.append(test_line)

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
        "malformed_rows_removed_first": len(malformed_rows),
        "prediction_lines_after_split_clean": len(split_clean_pred_lines),
        "invalid_aligned_rows_removed_second": len(deleted_aligned_rows),
        "invalid_placeholder_reply_texts": sorted(invalid_reply_set),
        "invalid_placeholder_rows_removed_second": sum(
            1 for row in deleted_aligned_rows
            if row.get("reason") == "invalid_placeholder_reply"
        ),
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
                "malformed_rows_removed_first": malformed_rows,
                "deleted_aligned_rows_removed_second": deleted_aligned_rows,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("============== 预测文件清洗完成 ==============")
    print(f"原始预测文件行数: {len(raw_pred_lines)}")
    print(f"第一步删除无法被 '{sep}' 精确划分成两部分的行数: {len(malformed_rows)}")
    print(f"第一步后预测文件行数: {len(split_clean_pred_lines)}")
    print(f"test 有效行数: {len(valid_test_lines)}")
    print(f"第二步同步删除 empty/json reply 行数: {len(deleted_aligned_rows)}")
    invalid_placeholder_count = sum(
        1 for row in deleted_aligned_rows
        if row.get("reason") == "invalid_placeholder_reply"
    )
    print(f"其中 API失败 reply 行数: {invalid_placeholder_count}")
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
            parts = line.rstrip("\n").split(sep)
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
