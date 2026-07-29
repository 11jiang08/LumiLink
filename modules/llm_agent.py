"""
LLM Agent 统一封装
==================
兼容 OpenAI 协议（DeepSeek / GPT-4o / 通义 等），支持：
- 普通对话补全
- 结构化 JSON 输出（Structured Output）
- Prompt 模板加载
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import LLM, PROMPTS_DIR

logger = logging.getLogger("LumiLink.llm")


class LLMAgent:
    """LLM 调用客户端单例。"""

    def __init__(self) -> None:
        if not LLM.api_key:
            logger.warning("LLM_API_KEY 未配置，LLM 调用将失败。请检查 .env 文件。")
        self.client = OpenAI(
            api_key=LLM.api_key or "dummy",
            base_url=LLM.base_url,
            timeout=LLM.timeout,
        )

    # ---------- 基础对话 ----------
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """普通文本补全。"""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=LLM.model,
            messages=messages,
            temperature=LLM.temperature if temperature is None else temperature,
            max_tokens=LLM.max_tokens if max_tokens is None else max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ---------- 结构化 JSON 输出 ----------
    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """要求模型返回 JSON 对象。若模型未原生支持 response_format，
        会做兜底解析（提取首个 ```json ... ``` 代码块或最外层花括号）。"""
        messages: list[dict[str, str]] = []
        sys_text = system or "你是一个严谨的 JSON 生成助手，只输出合法 JSON，不要任何额外说明。"
        messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=LLM.model,
                messages=messages,
                temperature=LLM.temperature if temperature is None else temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            # 兜底：不带 response_format 再请求一次，然后手动解析
            logger.warning("结构化 JSON 调用失败，尝试兜底解析：%s", e)
            text = self.chat(prompt, system=sys_text, temperature=temperature)
            return _extract_json(text)

    # ---------- Prompt 模板加载 ----------
    @staticmethod
    def load_prompt(name: str) -> str:
        """从 prompts/ 目录加载模板文件。"""
        path = PROMPTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Prompt 模板不存在: {path}")
        return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    """从自由文本中提取首个 JSON 对象。"""
    # 优先取 ```json ``` 代码块
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            return json.loads(text[start:end].strip())
    # 退而求其次，取最外层花括号
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return json.loads(text[start : end + 1])
    return {}


# 单例
_agent: LLMAgent | None = None


def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent
