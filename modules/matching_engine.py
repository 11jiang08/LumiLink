"""
LLM 逆向匹配引擎 (Matching Engine)
====================================
核心逻辑（通过 LLM Function Calling / Structured Output 实现）：
    1. 硬性排雷检测：A 的【缺点】是否命中 B 的【雷点】（反之亦然）。命中即否决。
    2. 软性互补计算：A 的【缺点】能否被 B 容纳或互补。
    3. 兴趣与场景交集：结合视觉识别出的当下场景与物品。

LLM 仅输出最佳匹配对象的 ID 与正向理由，缺点比对过程对用户隐藏。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import APP
from modules.llm_agent import get_agent
from modules.user_profile import MatchResult, UserProfile

logger = logging.getLogger("LumiLink.match")


class MatchingEngine:
    """基于 LLM 的逆向匹配引擎。"""

    def __init__(self) -> None:
        self._candidates: list[UserProfile] | None = None

    # ---------- 候选人加载 ----------
    def _load_candidates(self) -> list[UserProfile]:
        if self._candidates is not None:
            return self._candidates
        path: Path = APP.sample_users_path
        if not path.exists():
            logger.warning("样例用户文件不存在：%s，将使用空候选池。", path)
            self._candidates = []
            return self._candidates
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._candidates = [UserProfile.from_dict(item) for item in raw.get("users", [])]
        return self._candidates

    # ---------- 主流程 ----------
    def match(self, user_a: UserProfile) -> MatchResult:
        """为 user_a 在候选池中寻找最佳匹配。"""
        candidates = self._load_candidates()
        if not candidates:
            return MatchResult(
                reason="⚠️ 候选人池为空，请先在 data/sample_users.json 中配置假用户档案。"
            )

        prompt_template = get_agent().load_prompt(APP.matching_prompt_path.name)
        # 把所有候选人的完整档案（含缺点，仅 LLM 可见）拼成上下文
        candidates_block = "\n\n".join(
            f"--- 候选人 {c.user_id} ---\n{c.to_llm_text(reveal_weaknesses=True)}"
            for c in candidates
        )
        prompt = prompt_template.format(
            user_a=user_a.to_llm_text(reveal_weaknesses=True),
            candidates=candidates_block,
        )

        try:
            data = get_agent().chat_json(prompt)
        except Exception as e:
            logger.exception("LLM 匹配调用失败")
            return MatchResult(reason=f"⚠️ LLM 调用失败：{e}")

        return self._parse_result(data, candidates)

    # ---------- 结果解析 ----------
    @staticmethod
    def _parse_result(
        data: dict[str, Any], candidates: list[UserProfile]
    ) -> MatchResult:
        matched_id = str(data.get("matched_user_id", "")).strip()
        # 找到对应候选人，便于补全 nickname
        matched = next((c for c in candidates if c.user_id == matched_id), None)
        nickname = matched.nickname if matched else data.get("matched_nickname", "")

        return MatchResult(
            matched_user_id=matched_id,
            matched_nickname=nickname,
            compatibility_score=float(data.get("compatibility_score", 0.0)),
            avoid_mine_success=float(data.get("avoid_mine_success", 0.0)),
            reason=str(data.get("reason", "")),
            hidden_analysis=str(data.get("hidden_analysis", "")),
        )
