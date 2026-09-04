from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Tuple

import faiss
import numpy as np

from configuration import llm_config
from faiss_utils import save_faiss_index
from hf_model_utils import load_sentence_transformer

SEP = "-[split]-"


def _safe_name(app_id: str) -> str:
    """把 app_id 转成安全文件名。必须与 LogFaissRag._safe_name 规则一致。"""
    return str(app_id).replace("/", "_").replace("\\", "_").strip()


def _is_valid_date(date_text: str) -> bool:
    """日志日期只接受 YYYY-MM-DD。"""
    date_text = str(date_text or "").strip()
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _format_log_text(date_text: str, version: str, content: str) -> str:
    """生成用于 embedding 和 Prompt 的日志文本。"""
    date_text = str(date_text or "").strip()
    version = str(version or "").strip()
    content = str(content or "").strip()

    if date_text and version and content:
        return f"{date_text} {version} — {content}".strip()
    if date_text and content:
        return f"{date_text} — {content}".strip()
    return content


def _load_logs_from_single_file(
    log_file: str,
    *,
    require_valid_date: bool = True,
) -> Tuple[DefaultDict[str, List[str]], DefaultDict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    读取全量日志文件，并按 app_id 分桶。

    输入格式：
      app_id-[split]-date-[split]-version-[split]-content

    若 content 中误含分隔符，会用 SEP.join(terms[3:]) 拼回去，避免截断。
    """
    texts_by_app: DefaultDict[str, List[str]] = defaultdict(list)
    metas_by_app: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    stats: Dict[str, Any] = {
        "raw_lines": 0,
        "valid_lines": 0,
        "skipped_empty_lines": 0,
        "skipped_malformed_lines": 0,
        "skipped_missing_required_fields": 0,
        "skipped_invalid_date_lines": 0,
        "examples_malformed": [],
        "examples_invalid_date": [],
    }

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            stats["raw_lines"] += 1
            raw = line.rstrip("\n")

            if not raw.strip():
                stats["skipped_empty_lines"] += 1
                continue

            terms = raw.split(SEP)
            if len(terms) < 4:
                stats["skipped_malformed_lines"] += 1
                if len(stats["examples_malformed"]) < 5:
                    stats["examples_malformed"].append({"line_no": line_no, "line": raw})
                continue

            app_id = terms[0].strip()
            date_text = terms[1].strip()
            version = terms[2].strip()
            content = SEP.join(terms[3:]).strip()

            if not app_id or not date_text or not content:
                stats["skipped_missing_required_fields"] += 1
                continue

            if require_valid_date and not _is_valid_date(date_text):
                stats["skipped_invalid_date_lines"] += 1
                if len(stats["examples_invalid_date"]) < 5:
                    stats["examples_invalid_date"].append(
                        {"line_no": line_no, "date": date_text, "line": raw}
                    )
                continue

            text = _format_log_text(date_text, version, content)

            meta = {
                "app_id": app_id,
                "date": date_text,
                "version": version,
                "content": content,
                "text": text,
                "raw_line_no": line_no,
            }

            # 关键：texts_by_app[app_id][i] 与 metas_by_app[app_id][i]
            # 以及最终 FAISS index 中第 i 个向量严格一一对应。
            texts_by_app[app_id].append(text)
            metas_by_app[app_id].append(meta)
            stats["valid_lines"] += 1

    stats["apps_with_logs"] = len(texts_by_app)
    return texts_by_app, metas_by_app, stats


def _encode(model, texts: List[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def _resolve_output_dirs(args) -> tuple[str, str, str, str]:
    """
    解析输出目录。

    推荐只传 --out-dir，脚本会自动创建：
      out_dir/indexes
      out_dir/texts
      out_dir/metas

    若显式传入 --index-dir / --texts-dir / --metas-dir，则优先使用显式目录。
    """
    out_dir = str(getattr(args, "out_dir", "") or "").strip() or "./data/faiss/logs"
    index_dir = str(getattr(args, "index_dir", "") or "").strip() or os.path.join(out_dir, "indexes")
    texts_dir = str(getattr(args, "texts_dir", "") or "").strip() or os.path.join(out_dir, "texts")
    metas_dir = str(getattr(args, "metas_dir", "") or "").strip() or os.path.join(out_dir, "metas")
    return out_dir, index_dir, texts_dir, metas_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-app FAISS log indexes with separated metadata")
    parser.add_argument(
        "--log-file",
        "--logs-file",
        required=True,
        help="全量日志文件，格式：app_id-[split]-date-[split]-version-[split]-content",
    )
    parser.add_argument(
        "--out-dir",
        default="./data/faiss/logs",
        help="输出根目录，默认会在其中创建 indexes/texts/metas 三个子目录",
    )
    parser.add_argument("--index-dir", default="", help="可选：显式指定 FAISS index 目录")
    parser.add_argument("--texts-dir", default="", help="可选：显式指定 texts npy 目录")
    parser.add_argument("--metas-dir", default="", help="可选：显式指定 metas npy 目录")
    parser.add_argument("--model", default=llm_config.embedding_model, help="嵌入模型名称或本地路径")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-lines", type=int, default=1, help="每个 app 至少多少条日志才建索引")
    parser.add_argument(
        "--allow-invalid-date",
        action="store_true",
        help="允许日志日期不是 YYYY-MM-DD。通常不建议开启。",
    )
    args = parser.parse_args()

    log_file = str(args.log_file or "").strip()
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件不存在: {log_file}")

    out_dir, index_dir, texts_dir, metas_dir = _resolve_output_dirs(args)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(metas_dir, exist_ok=True)

    print("============== 输出目录 ==============")
    print("out_dir   =", os.path.abspath(out_dir))
    print("indexes   =", os.path.abspath(index_dir))
    print("texts     =", os.path.abspath(texts_dir))
    print("metas     =", os.path.abspath(metas_dir))

    print("============== 加载日志文件 ==============")
    print("log_file =", os.path.abspath(log_file))
    texts_by_app, metas_by_app, stats = _load_logs_from_single_file(
        log_file,
        require_valid_date=not bool(args.allow_invalid_date),
    )

    print("raw log lines =", stats["raw_lines"])
    print("valid log lines =", stats["valid_lines"])
    print("apps with logs =", stats["apps_with_logs"])
    print("skipped malformed =", stats["skipped_malformed_lines"])
    print("skipped invalid date =", stats["skipped_invalid_date_lines"])

    if not texts_by_app:
        raise ValueError(
            "没有读取到任何可用日志。请检查格式是否为："
            "app_id-[split]-date-[split]-version-[split]-content"
        )

    print("============== 加载 embedding 模型 ==============")
    print("model =", args.model)
    model = load_sentence_transformer(args.model)
    print("模型加载完成")

    total_apps = 0
    total_logs = 0
    built_apps: List[Dict[str, Any]] = []

    for app_id in sorted(texts_by_app.keys()):
        texts = texts_by_app[app_id]
        metas = metas_by_app[app_id]

        if len(texts) != len(metas):
            raise RuntimeError(f"texts/metas 数量不一致: {app_id} texts={len(texts)} metas={len(metas)}")

        if len(texts) < int(args.min_lines):
            print(f"[Skip] {app_id} | logs={len(texts)} < min_lines={args.min_lines}")
            continue

        safe_app = _safe_name(app_id)
        index_out = os.path.join(index_dir, f"{safe_app}.index")
        texts_out = os.path.join(texts_dir, f"{safe_app}.npy")
        metas_out = os.path.join(metas_dir, f"{safe_app}.npy")

        print(f"[Build] {app_id} | logs={len(texts)}")

        vecs = _encode(model, texts, args.batch_size)

        # 与评论索引脚本保持一致：主脚本直接 import faiss 并直接创建 IndexFlatIP。
        index = faiss.IndexFlatIP(int(vecs.shape[1]))
        index.add(vecs)

        save_faiss_index(index, index_out)
        np.save(texts_out, np.asarray(texts, dtype=object))
        np.save(metas_out, np.asarray(metas, dtype=object))

        total_apps += 1
        total_logs += len(texts)
        built_apps.append(
            {
                "app_id": app_id,
                "safe_app": safe_app,
                "logs": len(texts),
                "index_file": os.path.abspath(index_out),
                "texts_file": os.path.abspath(texts_out),
                "metas_file": os.path.abspath(metas_out),
            }
        )

    manifest = {
        "source_log_file": os.path.abspath(log_file),
        "out_dir": os.path.abspath(out_dir),
        "index_dir": os.path.abspath(index_dir),
        "texts_dir": os.path.abspath(texts_dir),
        "metas_dir": os.path.abspath(metas_dir),
        "file_naming": {
            "index": "indexes/<safe_app>.index",
            "text": "texts/<safe_app>.npy",
            "meta": "metas/<safe_app>.npy",
        },
        "model": str(args.model),
        "stats": stats,
        "apps_built": total_apps,
        "total_logs": total_logs,
        "built_apps": built_apps,
    }
    manifest_file = os.path.join(out_dir, "log_faiss_manifest.json")
    os.makedirs(os.path.dirname(os.path.abspath(manifest_file)), exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    print("=" * 60)
    print("完成")
    print("Apps built:", total_apps)
    print("Total logs:", total_logs)
    print("Index dir:", os.path.abspath(index_dir))
    print("Texts dir:", os.path.abspath(texts_dir))
    print("Metas dir:", os.path.abspath(metas_dir))
    print("Manifest:", os.path.abspath(manifest_file))


if __name__ == "__main__":
    main()
