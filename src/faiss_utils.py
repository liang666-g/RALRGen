"""
FAISS 索引构建 / 检索工具设置
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FaissIndexPaths:
    index_path: str


def _require_faiss():
    try:
        import faiss  # noqa: F401
    except ImportError as e:
        raise ImportError("请先安装 faiss-cpu：pip install faiss-cpu") from e


def save_faiss_index(index, index_path: str) -> None:
    _require_faiss()
    import faiss

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str):
    _require_faiss()
    import faiss

    return faiss.read_index(index_path)


def build_index_flat_ip(vectors: np.ndarray):
    """构建 IndexFlatIP（用于归一化向量的 cosine 相似）。"""
    _require_faiss()
    import faiss

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search_index(index, query_vecs: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (scores, ids)，形状均为 (nq, top_k)。"""
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype(np.float32)
    scores, ids = index.search(query_vecs, int(top_k))  # 在FAISS索引里检索相似评论，返回前k个最相似结果的相似度分数和命中的向量id
    return scores, ids

