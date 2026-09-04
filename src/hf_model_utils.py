from __future__ import annotations

from typing import Optional
from configuration import llm_config


def load_sentence_transformer(model_name: str, device: Optional[str] = None):
    """
    统一加载 SentenceTransformer。

    作用：
    1. 优先使用 configuration.py 中的本地模型路径
    2. 使用本地 cache_folder
    3. 如果配置 hf_local_files_only=True，则尽量禁止访问 HuggingFace 线上
    4. 兼容旧版本 sentence-transformers 不支持 local_files_only 的情况
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "请先安装 sentence-transformers：pip install sentence-transformers"
        ) from e

    kwargs = {
        "cache_folder": getattr(llm_config, "model_cache_dir", None),
    }

    if device is not None:
        kwargs["device"] = device

    if bool(getattr(llm_config, "hf_local_files_only", False)):
        kwargs["local_files_only"] = True

    try:
        return SentenceTransformer(model_name, **kwargs)
    except TypeError:
        # 兼容旧版 sentence-transformers 不支持 local_files_only 参数
        kwargs.pop("local_files_only", None)
        return SentenceTransformer(model_name, **kwargs)
        