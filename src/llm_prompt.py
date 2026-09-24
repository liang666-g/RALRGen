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
    # extra = _block_extra(extra_review)
    # sentiment = str(sentiment_info).strip()
    review = user_review.strip() or "(The user review is empty.)"
    logs = _format_simi_block(log_texts) if log_texts is not None else "(No retrieved release notes.)"
    context_prompt = f"""
You are a senior official customer service expert for an app. Your task is to draft a fully English response that can be sent directly to the user, based on the provided information.

[Structured Reasoning]
Before writing the final reply, internally follow these steps:
1. Identify the user's main issue and key concern.
2. Examine the provided historical examples, release logs, and app feature phrases for relevant information.
3. Assess which retrieved information is applicable to the current review, and ignore irrelevant or unsupported context.
4. Plan a concise response that addresses the issue and follows the response strategy and style of the most relevant historical example.

[Reply Rules]
Use the most relevant historical reply as a reference for response strategy and style, and adapt it to the current review.
Use release-log information only when it is directly relevant to the user's issue.
Treat app feature phrases only as general product background, rather than evidence of the exact software state at the time of the review.
Do not introduce unsupported causes, fixes, features, instructions, dates, or version information.
Keep the response concise and natural, with wording and length reasonably close to the most relevant historical official reply.

[Target Review]
{review}

[Historical Reply Examples] 
{rag}

[Relevant Release Logs]
{logs}

[App Feature Phrases]
{features}
"""
    cot = getattr(llm_config, "cot_json_mode", True)
    if cot:
        system = context_prompt
    else:
        system = f"""
You are a senior official customer service expert for an app. Your task is to draft a fully English response that can be sent directly to the user, based on the provided information.

[Reply Rules]
Use the most relevant historical reply as a reference for response strategy and style, and adapt it to the current review.
Use release-log information only when it is directly relevant to the user's issue.
Treat app feature phrases only as general product background, rather than evidence of the exact software state at the time of the review.
Do not introduce unsupported causes, fixes, features, instructions, dates, or version information.
Keep the response concise and natural, with wording and length reasonably close to the most relevant historical official reply.

[Target Review]
{review}

[Historical Reply Examples] 
{rag}

[Relevant Release Logs]
{logs}

[App Feature Phrases]
{features}
"""
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": llm_config.User_role_prompt},
    ]
