from __future__ import annotations
import argparse
import os
import re
from collections import defaultdict
from configuration import llm_config
import faiss
import numpy as np
from faiss_utils import save_faiss_index
from hf_model_utils import load_sentence_transformer


def _encode(model, texts, batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def _safe_name(app_id: str) -> str:
    return str(app_id).replace("/", "_").replace("\\", "_").strip()


def _safe_rating(rating: str) -> str:
    r = str(rating or "").strip()
    if not r:
        return "unknown"
    return r.replace("/", "_").replace("\\", "_").replace(" ", "_")


def main():
    p = argparse.ArgumentParser(description="Build per-app FAISS indexes for train comments")
    p.add_argument("--train-file", required=True, help="训练文件")
    p.add_argument("--index-dir", required=True, help="输出索引目录")
    p.add_argument("--offsets-dir", required=True, help="输出 offsets 目录")
    p.add_argument("--model", default=llm_config.embedding_model, help="嵌入模型名称")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="至少多少条评论才为该 app 建索引，默认 1",
    )
    args = p.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)
    os.makedirs(args.offsets_dir, exist_ok=True)

    # 按 app_id 分桶
    texts_by_bucket = defaultdict(list)
    offsets_by_bucket = defaultdict(list)

    with open(args.train_file, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                s = line.decode("utf-8")
            except UnicodeDecodeError:
                s = line.decode("utf-8", errors="replace")

            terms = s.split("-[split]-")
            if len(terms) < 8:
                continue

            app_id = terms[0].strip()
            rating = terms[1].strip()
            src_sent = re.sub(r"[.,!?<>():;\[\]]", " ", terms[4]).strip()

            if not app_id or not rating or not src_sent:
                continue
            
            # 1. 当前 App + 当前 rating 桶
            rating_bucket_key = (app_id, rating)
            texts_by_bucket[rating_bucket_key].append(src_sent)
            offsets_by_bucket[rating_bucket_key].append(pos)

            # 2. 当前 App + all 桶，用于 rating 缺失时回退
            all_bucket_key = (app_id, "all")
            texts_by_bucket[all_bucket_key].append(src_sent)
            offsets_by_bucket[all_bucket_key].append(pos)

    model = load_sentence_transformer(args.model)

    total_buckets = 0
    total_vectors = 0
    apps_built = set()

    for (app_id, rating), texts in texts_by_bucket.items():
        if len(texts) < int(args.min_samples):
            continue

        safe_app = _safe_name(app_id)
        safe_rating = _safe_rating(rating)

        app_index_dir = os.path.join(args.index_dir, safe_app)
        app_offsets_dir = os.path.join(args.offsets_dir, safe_app)

        os.makedirs(app_index_dir, exist_ok=True)
        os.makedirs(app_offsets_dir, exist_ok=True)

        index_out = os.path.join(app_index_dir, f"{safe_rating}.index")
        offsets_out = os.path.join(app_offsets_dir, f"{safe_rating}_offsets.npy")

        print(f"[Build] {app_id} | rating={rating} | comments={len(texts)}")

        vecs = _encode(model, texts, args.batch_size)
        index = faiss.IndexFlatIP(int(vecs.shape[1]))
        index.add(vecs)

        save_faiss_index(index, index_out)
        np.save(offsets_out, np.asarray(offsets_by_bucket[(app_id, rating)], dtype=np.int64))

        total_buckets += 1
        total_vectors += int(index.ntotal)
        apps_built.add(app_id)

    print("=" * 60)
    print("完成")
    print("Apps built:", len(apps_built))
    print("App-rating buckets built:", total_buckets)
    print("Total vectors:", total_vectors)


if __name__ == "__main__":
    main()