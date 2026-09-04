"""
App 更新日志的语义检索（RAG）。按 app_id 找到对应日志文件，把日志文本在线编码成向量，再拿当前评论去做相似度匹配，取 top-k 条最相关日志。不使用 FAISS

数据来源：``data/app_logs/logs_predict_ds/<app_id>.txt``
每行格式示例：
  2016-04-04-[split]-2.128.3-[split]-Faster load times (less data usage)

推理时对「当前用户评论」做向量检索，返回 top-k 条最相关日志片段，写入提示词。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from configuration import llm_config
import numpy as np


@dataclass(frozen=True)
class LogRagConfig:
    logs_dir: str
    model_name: str = llm_config.embedding_model
    top_k: int = 3
    max_lines_per_app: Optional[int] = 2000
    device: Optional[str] = None


"""
构建缓存，同一个 app 的日志，不要每次检索都重新读取和重新向量化
第一次检索某个 app 的日志时
→ 读文件、编码成向量、存进缓存
以后再检索这个 app
→ 直接从缓存里拿，不再重复计算
"""
_CACHE: Dict[Tuple[str, str], Tuple[List[str], np.ndarray]] = {}


# 把日志文件里的每一行解析成可检索文本
def _parse_log_line(line: str) -> str:
    parts = line.strip().split("-[split]-")
    if len(parts) >= 3:
        date, version, content = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if date and version:
            return f"{date} {version} — {content}".strip()
        return content.strip()
    return line.strip()


# 读取该 app 的日志文本列表
def _load_log_texts(app_id: str, logs_dir: str, max_lines: Optional[int]) -> List[str]:
    fp = os.path.join(logs_dir, f"{app_id}.txt")
    if not os.path.exists(fp):
        return []
    texts = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= int(max_lines):
                break
            t = _parse_log_line(line)
            if t:
                texts.append(t)
    return texts


def _get_model(model_name: str, device: Optional[str]):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("日志语义检索需要 sentence-transformers。请先安装: pip install -r requirements-retrieval.txt")
    return SentenceTransformer(model_name, device=device)


def retrieve_relevant_logs(app_id: str, query: str, cfg: LogRagConfig,) -> List[str]:
    """返回与 query 最相关的 top-k 日志片段（字符串列表）。"""
    query = (query or "").strip()
    if not query:
        return []

    key = (cfg.logs_dir, app_id)  # 构造缓存键，将日志目录和app_id一起作为唯一键
    cached = _CACHE.get(key)  # 检查当前cache中是否存在当前k键，存在就取出，不存在返回None
    if cached is None:
        texts = _load_log_texts(app_id, cfg.logs_dir, cfg.max_lines_per_app)
        if not texts:  # 若 app 没有日志，缓存空结果
            _CACHE[key] = ([], np.zeros((0, 1), dtype=np.float32))
            return []
        model = _get_model(cfg.model_name, cfg.device)
        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )  # 日志文本向量化
        mat = np.asarray(emb, dtype=np.float32)
        _CACHE[key] = (texts, mat)  # 将日志文本和向量化矩阵存入缓存中，下次对同一个App检索日志时可直接取出使用
        cached = (texts, mat)

    # 取出日志文本和向量化矩阵
    texts, mat = cached
    if not texts or mat.size == 0:
        return []

    # 向量化查询评论 
    model = _get_model(cfg.model_name, cfg.device)
    q = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)
    sims = mat @ q  # 用矩阵乘法计算相似度，拿当前评论和该 app 的所有日志逐条做语义相似度比较

    want = min(int(cfg.top_k), int(sims.shape[0]))
    if want <= 0:
        return []
    if sims.shape[0] <= want:  # 日志数量<k，全取
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, want - 1)[:want]  # 先找出前k个相似日志
        top_idx = top_idx[np.argsort(-sims[top_idx])]  # 精确排序
    return [texts[int(i)] for i in top_idx[:want]]

