"""
为大模型构造 system / user 消息（含 RAG 检索范例）。
"""
from __future__ import annotations
from typing import List, Union
from configuration import llm_config


def _format_desc(feature_phrases: str) -> str:
    text = str(feature_phrases or "").strip()
    if not text:
        return "(No app feature phrases are available.)"
    return text


def _format_simi_block(simi: Union[str, List[str]]) -> str:
    if isinstance(simi, str):
        t = simi.strip()
        return t if t else "(No examples found.)"
    lines = []
    for i, item in enumerate(simi or []):
        if isinstance(item, dict):
            src = item.get("src_review", "")
            tgt = item.get("tgt_reply", "")
            if src and tgt:
                lines.append(f"example {i + 1}:\nsimilar comments:{src}\nofficial response:{tgt}")
            elif tgt:
                lines.append(f"example {i + 1}:\nofficial response:{tgt}")
            elif src:
                lines.append(f"example {i + 1}:\nsimilar comments:{src}")
            # if src and tgt:
            #     lines.append(f"example {i + 1}:{tgt}")
            # elif tgt:
            #     lines.append(f"example {i + 1}:{tgt}")
            # elif src:
            #     lines.append(f"example {i + 1}:\nsimilar comments:{src}")
        else:
            t = str(item or "")
            if t:
                lines.append(f"example {i + 1}:{t}")

    return "\n\n".join(lines) if lines else "(No examples found.)"


def _block_extra(extra_review: str) -> str:
    t = extra_review.strip()
    return t if t else "(No additional similar user comments.)"


def build_messages(
    user_review: str,
    feature_phrases: str,
    simi_texts: Union[str, List[str]],
    extra_review: str,
    log_texts: Union[None, str, List[str]] = None,
    sentiment_info: str = "",
) -> List[dict]:
    # 构建 messages 列表（system + user）。
    features = _format_desc(feature_phrases)
    rag = _format_simi_block(simi_texts)
    extra = _block_extra(extra_review)
    sentiment = str(sentiment_info).strip()
    review = user_review.strip() or "(The user review is empty.)"
    logs = _format_simi_block(log_texts) if log_texts is not None else "(No retrieved release notes.)"
    user = f"""
Generate an official customer-service reply using the context below.
[current user comments]
{review}

[historical reply examples] 
{rag}

[related logs]
{logs}

[App feature phrases]
{features}
"""
    cot = getattr(llm_config, "cot_json_mode", True)
    if cot:
        system = getattr(llm_config, "system_prompt", None)
    else:
        system = f"""
Please directly output a customer service reply that can be sent directly to the user (do not output the thought process).
"""
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
