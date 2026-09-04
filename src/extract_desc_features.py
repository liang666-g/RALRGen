#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract user-facing App feature phrases from description text files.

Default input mode:
  ./data/app_descriptions/<app_name>.txt

Default output format:
  app_name-[split]-feature phrase ; feature phrase ; ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import requests

try:
    import local_api_config as local_api
except ImportError:
    local_api = None


def _local_attr(name: str, default=None):
    if local_api is None:
        return default
    return getattr(local_api, name, default)


def _as_bool(value, default: bool = False) -> bool:
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


def _default_api_key_pool() -> List[str]:
    pool = _local_attr("OPENAI_API_KEY_POOL", []) or []
    if isinstance(pool, str):
        keys = [x.strip() for x in pool.split(",") if x.strip()]
    else:
        keys = [str(x).strip() for x in pool if str(x).strip()]

    single_key = str(_local_attr("OPENAI_API_KEY", "") or "").strip()
    if single_key:
        keys.append(single_key)

    if not keys:
        keys = [
            x.strip()
            for x in os.environ.get("OPENAI_API_KEY_POOL", "").split(",")
            if x.strip()
        ]
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_key:
            keys.append(env_key)

    return keys


def _default_model() -> str:
    return str(
        _local_attr("OPENAI_MODEL", "")
        or os.environ.get("OPENAI_MODEL", "")
        or "deepseek-v4-flash"
    ).strip()


def _default_base_url() -> str:
    return str(
        _local_attr("OPENAI_BASE_URL", "")
        or os.environ.get("OPENAI_BASE_URL", "")
        or "https://api.deepseek.com"
    ).strip()


def _default_stream() -> bool:
    local_value = _local_attr("OPENAI_STREAM", None)
    if local_value is not None:
        return _as_bool(local_value, default=False)
    return _as_bool(os.environ.get("OPENAI_STREAM"), default=False)


class DeepSeekFeatureExtractor:
    """Extract App feature phrases with an OpenAI-compatible DeepSeek API."""

    def __init__(
        self,
        api_keys: List[str],
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        stream: bool = False,
        max_tokens: int = 4096,
        timeout: int = 60,
        retries: int = 2,
    ) -> None:
        self.api_keys = [key.strip() for key in api_keys if key and key.strip()]
        self.model = model
        self.base_url = base_url
        self.stream = stream
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def _api_key(self) -> str:
        if not self.api_keys:
            raise ValueError("No API key found. Set OPENAI_API_KEY_POOL in local_api_config.py.")
        return random.choice(self.api_keys)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _stream_content(response) -> str:
        chunks: List[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                chunks.append(text)
        return "".join(chunks)

    def build_messages(self, desc_text: str) -> List[dict]:
        system_prompt = (
            "You are a professional mobile app product analysis assistant."
            "Your task is to extract key feature phrases from app description text."
            "You must strictly adhere to the extraction rules provided by the user and output only JSON."
        )

        user_prompt = f"""
# Task
Extract the core user-facing feature phrases from the App description below.
The extracted phrases will be used only as optional product background for customer-service reply generation.

# Extraction Rules
1.Extract only functions that are explicitly supported by the source text.
2.A valid feature should describe something the user can do with the App, or a concrete capability, service, module, page, or product mechanism.
3.Prefer concise and natural English noun phrases, usually 2 to 6 words.
4.Keep only independently meaningful core features.
5.Merge phrases that describe the same function or substantially overlapping functions. Keep the clearer or more specific phrase.
6.Do not infer a function from the App category, brand, popularity, or general product knowledge.
7.Do not extract:
    - advertising slogans;
    - vague advantages or overall evaluations;
    - popularity or user-base claims;
    - quality, safety, or reliability claims;
    - prices, discounts, free offers, trials, rewards, or promotional policies;
    - broad phrases such as "basic features", "easy to use", or "high quality";
    - details that only describe the company or business model.
8. Authentication methods, subscription plans, payment methods, and account
   options should be extracted only when they are clearly presented as an
   important user-facing capability, not merely as setup or commercial details.
9. Order the retained features by their importance to the App's primary purpose.
10. Before producing the JSON, internally verify every phrase:
    - Is it explicitly supported by the description?
    - Is it a concrete user-facing capability?
    - Is it meaningfully different from the other phrases?
    Remove the phrase if any answer is no.

# Example
description:
uber be a ridesharing app for request ride track driver pay with card or cash and receive receipt by email

Output:
{{
  "feature_phrases": [
    "ride booking",
    "driver tracking",
    "card payment",
    "cash payment",
    "trip receipt by email"
  ]
}}

# Output Format
Output exactly one JSON object.
Do not output explanations, prefixes, suffixes, or Markdown code fences.

# App Description
{desc_text}
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def extract_feature_phrases(self, desc_text: str, timeout: int | None = None) -> List[str]:
        messages = self.build_messages(desc_text)
        request_timeout = self.timeout if timeout is None else timeout

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "stream": self.stream,
        }

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    self._chat_completions_url(),
                    headers={
                        "Authorization": f"Bearer {self._api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                content = str(content or "")
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start < 0 or json_end <= json_start:
                    print(f"[WARN] Could not extract JSON from response: {content[:200]}")
                    return []

                obj = json.loads(content[json_start:json_end])
                phrases = obj.get("feature_phrases", [])

                if not isinstance(phrases, list):
                    return []

                cleaned: List[str] = []
                seen = set()
                for phrase in phrases:
                    if not isinstance(phrase, str):
                        continue
                    phrase = phrase.strip()
                    if not phrase:
                        continue
                    key = phrase.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cleaned.append(phrase)

                return cleaned[:10]

            except json.JSONDecodeError as exc:
                print(f"[ERROR] JSON parse failed: {exc}")
                return []
            except Exception as exc:
                if attempt < self.retries:
                    wait_seconds = min(2 ** attempt, 8)
                    print(
                        f"[WARN] API call failed on attempt {attempt + 1}/{self.retries + 1}: {exc}. "
                        f"Retrying in {wait_seconds}s."
                    )
                    time.sleep(wait_seconds)
                    continue
                print(f"[ERROR] API call failed: {exc}")
                return []

        return []

    def batch_extract(
        self,
        items: List[Tuple[str, str]],
        delay: float = 1.0,
    ) -> List[Tuple[str, List[str]]]:
        results: List[Tuple[str, List[str]]] = []

        for index, (app_name, desc_text) in enumerate(items):
            print(f"Processing {index + 1}/{len(items)}: {app_name}")
            if not desc_text.strip():
                results.append((app_name, []))
                continue

            phrases = self.extract_feature_phrases(desc_text)
            results.append((app_name, phrases))

            if index < len(items) - 1 and delay > 0:
                time.sleep(delay)

        return results


def parse_desc_line(line: str) -> Tuple[str, str]:
    s = line.strip()
    if not s:
        return "", ""

    parts = s.split("-[split]-", 1)
    if len(parts) != 2:
        return "", ""

    app_id = parts[0].strip()
    desc_text = parts[1].strip()
    return app_id, desc_text


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="replace")


def load_description_files(input_dir: str) -> List[Tuple[str, str]]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Description directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Description input is not a directory: {root}")

    items: List[Tuple[str, str]] = []
    for path in sorted(root.glob("*.txt")):
        app_name = path.stem.strip()
        if not app_name:
            continue
        desc_text = read_text_file(path).strip()
        items.append((app_name, desc_text))

    return items


def load_legacy_desc_file(input_path: str) -> List[Tuple[str, str]]:
    try:
        with open(input_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except UnicodeDecodeError:
        with open(input_path, "r", encoding="gbk") as handle:
            lines = handle.readlines()

    items: List[Tuple[str, str]] = []
    for line in lines:
        app_id, desc_text = parse_desc_line(line)
        if app_id and desc_text:
            items.append((app_id, desc_text))
    return items


def filter_items_by_app_names(items: List[Tuple[str, str]], app_names: str) -> List[Tuple[str, str]]:
    names = {name.strip().lower() for name in app_names.split(",") if name.strip()}
    if not names:
        return items
    return [(app_name, desc_text) for app_name, desc_text in items if app_name.lower() in names]


def write_jsonl_output(output_path: str, results: List[Tuple[str, List[str]]]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for app_name, phrases in results:
            obj = {
                "app_id": app_name,
                "feature_phrases": phrases,
            }
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_desc_style_output(output_path: str, results: List[Tuple[str, List[str]]]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for app_name, phrases in results:
            phrase_str = " ; ".join(phrases)
            handle.write(f"{app_name}-[split]-{phrase_str}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract App feature phrases from description txt files.")
    parser.add_argument(
        "--input_dir",
        default="./data/app_descriptions",
        help="Directory containing one .txt description file per app.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional legacy desc file path: app_id-[split]-description.",
    )
    parser.add_argument("--output", default="./data/desc_features.txt", help="Output file path.")
    parser.add_argument("--api_key", "-k", default="", help="Optional single OpenAI-compatible API key override.")
    parser.add_argument("--model", default=_default_model(), help="OpenAI-compatible model name.")
    parser.add_argument("--base_url", default=_default_base_url(), help="OpenAI-compatible base URL.")
    parser.add_argument("--stream", action="store_true", default=_default_stream(), help="Use streaming output if enabled.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum completion tokens for feature extraction.")
    parser.add_argument("--timeout", type=int, default=90, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for transient API failures.")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Request interval in seconds.")
    parser.add_argument("--max_items", type=int, default=0, help="Maximum number of files/items to process; 0 means all.")
    parser.add_argument("--apps", default="", help="Optional comma-separated app names to process.")
    parser.add_argument("--output_format", choices=["jsonl", "desc"], default="desc", help="Output format: jsonl or desc.")
    args = parser.parse_args()

    api_keys = [args.api_key.strip()] if args.api_key.strip() else _default_api_key_pool()
    if not api_keys:
        raise ValueError("No API key found. Set OPENAI_API_KEY_POOL in local_api_config.py or pass --api_key.")

    if args.input:
        items = load_legacy_desc_file(args.input)
    else:
        items = load_description_files(args.input_dir)

    items = filter_items_by_app_names(items, args.apps)

    if args.max_items > 0:
        items = items[: args.max_items]

    print(f"Loaded {len(items)} app description item(s).")

    extractor = DeepSeekFeatureExtractor(
        api_keys=api_keys,
        model=args.model,
        base_url=args.base_url,
        stream=args.stream,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )
    results = extractor.batch_extract(items, delay=args.delay)

    if args.output_format == "jsonl":
        write_jsonl_output(args.output, results)
    else:
        write_desc_style_output(args.output, results)

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
