from __future__ import annotations
import json
import os
import re
import time
from http import HTTPStatus
from typing import Any, Dict, List, Tuple
from configuration import llm_config
import dashscope
from dashscope import Generation
from openai import OpenAI
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


def strip_code_fence(text: str) -> str:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*\r?\n?(.*)\r?\n?```\s*$", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _messages_to_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip() or "user"
        content = str(msg.get("content", ""))
        converted.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": content,
                    }
                ],
            }
        )
    return converted


def parse_cot_json_output(raw: str) -> Tuple[str, str]:
    """
    从模型输出中解析思维链 JSON：字段 thoughts（前 3 步合并叙述）、reply（最终英文回复）。

    若解析失败，返回 (原始文本截断, "")，由调用方决定是否降级。
    """
    s = strip_code_fence(raw)
    start = s.find("{")
    if start == -1:
        return (raw.strip()[:2000], "")

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s, start)
        thoughts = str(obj.get("thoughts", ""))
        reply = str(obj.get("reply", ""))
        return thoughts.strip(), reply.strip()
    except json.JSONDecodeError:
        pass

    thoughts = ""
    reply = ""
    
    # 提取 thoughts 内容：匹配 "thoughts": " 到下一个 ,"reply" 之间的所有内容
    t_match = re.search(r'"thoughts"\s*:\s*"(.*?)"\s*,\s*"reply"', s, re.DOTALL)
    if not t_match:
        t_match = re.search(r'"thoughts"\s*:\s*"(.*?)"\s*}', s, re.DOTALL)
    if t_match:
        thoughts = t_match.group(1)
        
    # 提取 reply 内容：匹配 "reply": " 到结束大括号 } 之间的所有内容
    r_match = re.search(r'"reply"\s*:\s*"(.*?)"\s*}', s, re.DOTALL)
    if not r_match:
        r_match = re.search(r'"reply"\s*:\s*"(.*?)"\s*,', s, re.DOTALL)
    if r_match:
        reply = r_match.group(1)

    # 3. 如果正则提取到了内容，手动清理转义符并返回
    if thoughts or reply:
        thoughts = thoughts.replace('\\n', '\n').replace('\\"', '"')
        reply = reply.replace('\\n', '\n').replace('\\"', '"')
        return thoughts.strip(), reply.strip()

    # 4. 如果连正则都失败了，才返回原始文本让上层降级
    return (raw.strip()[:2000], "")


def _decode_json_string_fragment(value: str) -> str:
    """尽量把 JSON 字符串片段中的常见转义还原为普通文本。"""
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return (
            value
            .replace('\\n', ' ')
            .replace('\\r', ' ')
            .replace('\\t', ' ')
            .replace('\\"', '"')
            .replace('\\\\', '\\')
        )


def clean_plain_reply_output(raw: str) -> str:
    """
    清洗纯文本回复模式下的模型输出。

    目标：
    1. 正常情况下直接返回模型生成的英文客服回复。
    2. 如果模型仍误输出完整 JSON，则优先提取 reply 字段。
    3. 如果模型输出半截 JSON，尽量用正则抢救 reply 字段，避免整段 JSON 写入 predictions.txt。
    4. 统一压成单行，避免预测文件按行评测时错位。
    """
    s = strip_code_fence(raw or "").strip()
    if not s:
        return ""

    # 兼容模型偶尔仍然输出 JSON 的情况：完整 JSON 优先用 json 解析。
    json_start = s.find("{")
    if json_start != -1 and (s.lstrip().startswith("{") or '"reply"' in s):
        try:
            obj, _ = json.JSONDecoder().raw_decode(s, json_start)
            if isinstance(obj, dict) and str(obj.get("reply", "")).strip():
                s = str(obj.get("reply", ""))
        except json.JSONDecodeError:
            # 抢救半截 JSON 中的 reply 字段：先匹配完整字符串，再匹配到文本末尾。
            m = re.search(r'"reply"\s*:\s*"((?:\\.|[^"\\])*)"', s, flags=re.DOTALL)
            if not m:
                m = re.search(r'"reply"\s*:\s*"(.*)$', s, flags=re.DOTALL)
            if m:
                s = _decode_json_string_fragment(m.group(1))
            else:
                # 没有可用 reply 时，不把半截 JSON 当作回复写入评测文件。
                return ""

    # 去掉常见标签，防止写成 "Reply: ..."。
    s = re.sub(
        r'^\s*(?:here\s+is\s+(?:the\s+)?(?:reply|response)\s*[:：]?\s*|final\s+reply|reply|response|customer\s+reply|客服回复|最终回复)\s*[:：]\s*',
        '',
        s,
        flags=re.IGNORECASE,
    )

    # 去掉整体包裹引号。
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()

    # 评测文件是按行写入的，必须压成单行；同时避免与 predictions.txt 的 "**" 分隔符冲突。
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r'\s*\n+\s*', ' ', s)
    s = re.sub(r'[ \t]+', ' ', s).strip()
    s = s.replace("-[split]-", " ")
    return s


# 从 Dashscope 响应对象里，把真正的文本内容取出来
def _extract_message_text(resp: Any) -> str:
    out = getattr(resp, "output", None)
    if out is None:
        raise RuntimeError(f"API 无 output: {resp}")
    if isinstance(out, dict):
        if out.get("text"):
            return str(out["text"]).strip()
        choices = out.get("choices") or []
    else:
        if getattr(out, "text", None):
            return str(out.text).strip()
        choices = getattr(out, "choices", None) or []
    if not choices:
        raise RuntimeError(f"无法解析模型输出: {out}")
    first = choices[0]
    if isinstance(first, dict):
        msg = first.get("message") or {}
        content = msg.get("content", "")
    else:
        msg = getattr(first, "message", None)
        content = getattr(msg, "content", "") if msg is not None else ""
    return str(content).strip()


def _extract_openai_response_text(resp: Any) -> str:
    """
    提取 OpenAI / DeepSeek chat.completions.create 返回中的文本内容。
    兼容对象形式与 dict 形式。
    """
    # dict 形式
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI/DeepSeek 响应中无 choices: {resp}")

        first = choices[0]
        # print(first)

        msg = first.get("message") or {}
        content = msg.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text", "")
                else:
                    txt = getattr(item, "text", "")
                if txt:
                    parts.append(str(txt).strip())
            return "\n".join(parts).strip()

        return str(content).strip()

    else:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise RuntimeError(f"OpenAI/DeepSeek 响应中无 choices: {resp}")

        first = choices[0]
        # print(first)

        msg = getattr(first, "message", None)
        if msg is None:
            raise RuntimeError(f"OpenAI/DeepSeek 响应中无 message: {resp}")

        content = getattr(msg, "content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                txt = getattr(item, "text", "")
                if txt:
                    parts.append(str(txt).strip())
            return "\n".join(parts).strip()

        return str(content).strip()


def _extract_openai_stream_text(stream: Any) -> str:
    """
    Read text from an OpenAI-compatible chat completion stream.

    chat.completions.create(..., stream=True) returns an iterable Stream object,
    where each chunk carries incremental text in choices[*].delta.content.
    """
    parts: List[str] = []

    for chunk in stream:
        if isinstance(chunk, dict):
            choices = chunk.get("choices") or []
        else:
            choices = getattr(chunk, "choices", None) or []

        for choice in choices:
            if isinstance(choice, dict):
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
            else:
                delta = getattr(choice, "delta", None)
                message = getattr(choice, "message", None)

            content = ""
            if isinstance(delta, dict):
                content = delta.get("content") or ""
            elif delta is not None:
                content = getattr(delta, "content", "") or ""

            # Some OpenAI-compatible providers may put the final content on
            # message instead of delta in the last chunk.
            if not content:
                if isinstance(message, dict):
                    content = message.get("content") or ""
                elif message is not None:
                    content = getattr(message, "content", "") or ""

            if isinstance(content, str):
                if content:
                    parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        txt = item.get("text") or ""
                    else:
                        txt = getattr(item, "text", "") or ""
                    if txt:
                        parts.append(str(txt))
            elif content:
                parts.append(str(content))

    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(f"OpenAI/DeepSeek 流式响应中无 content: {stream}")
    return text


def _is_openai_stream_response(resp: Any) -> bool:
    """
    Detect whether an OpenAI-compatible response is a streaming response.

    Non-stream chat completion responses expose choices directly. Streaming
    responses are iterable Stream objects whose chunks expose choices.
    """
    if isinstance(resp, dict):
        return False
    if getattr(resp, "choices", None) is not None:
        return False
    return hasattr(resp, "__iter__")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True # 如果 3 次都失败了，就把异常抛出去给上层
)   

def generate_LLM(messages: List[Dict[str, str]]) -> str:
    """
    调用大模型生成（DashScope / OpenAI），返回纯文本。

    需在环境中设置 llm_config.api_key_env 所命名的变量：
    - DashScope: DASHSCOPE_API_KEY
    """
    provider = str(getattr(llm_config, "provider", "openai")).strip().lower()
    key_pool = getattr(llm_config, "api_key_pool", [])
    if key_pool:
        key = random.choice(key_pool)
    else:
        key = os.environ.get(llm_config.api_key_env, "").strip()
        
    if not key:
        raise ValueError(
            f"未检测到 API Key，请先设置环境变量 {llm_config.api_key_env}"
        )

    if provider == "dashscope":
        dashscope.api_key = key
        kwargs = dict(
            model=llm_config.model,
            messages=messages,
            temperature=float(llm_config.temperature),  # 采样随机性
            result_format="message",  # 返回格式为message   
        )
        if getattr(llm_config, "max_tokens", None):  # 限制最大生成长度
            kwargs["max_tokens"] = int(llm_config.max_tokens)

        resp = Generation.call(**kwargs)  # 调用千问API
        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            code = getattr(resp, "code", "")
            msg = getattr(resp, "message", "")
            raise RuntimeError(
                f"DashScope 调用失败: status={getattr(resp, 'status_code', None)} code={code} message={msg}"
            )
        text = _extract_message_text(resp)

    elif provider == "openai":
        client = OpenAI(
            api_key=key,
            base_url=str(getattr(llm_config, "openai_base_url", "") or "").strip() or None,
        )
        use_stream = _as_bool(
            getattr(
                llm_config,
                "stream",
                getattr(llm_config, "stream_output", True),
            ),
            default=True,
        )
        kwargs = dict(
            model=str(llm_config.model),
            messages=messages,
            temperature=float(llm_config.temperature),
            stream=use_stream,
        )
        if getattr(llm_config, "max_tokens", None):
            kwargs["max_tokens"] = int(llm_config.max_tokens)

        resp = client.chat.completions.create(**kwargs)
        if _is_openai_stream_response(resp):
            text = _extract_openai_stream_text(resp)
        else:
            text = _extract_openai_response_text(resp)

    else:
        raise ValueError(f"未知 llm_config.provider={provider!r}，请设置为 'dashscope' 或 'openai'")

    text = str(text or "").strip()
    if not text:
        raise RuntimeError("LLM API 返回空 content")

    interval = float(getattr(llm_config, "request_interval_sec", 0) or 0)  # 请求间隔时间
    if interval > 0:
        time.sleep(interval)
    return text


def generate_cot(messages: List[Dict[str, str]]) -> Tuple[str, str, str]:
    raw = generate_LLM(messages)
    reply = clean_plain_reply_output(raw)
    if not (reply or "").strip():
        raise RuntimeError("LLM 输出经过清洗后 reply 为空")
    thoughts = ""
    return raw, thoughts, reply
