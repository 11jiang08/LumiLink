"""
高情商破冰与线下行动引擎 (Action Engine)
==========================================
目标：解决「见光死」与「尬聊」痛点。

输出《专属线下破冰行动指南》，包含：
    1. 三种风格破冰开场白（直球型 / 幽默型 / 社恐专属型）
    2. 破冰场所建议（结合双方性格与缺点，如社恐推荐猫咖）
    3. 话题避坑指南（根据对方雷点生成建议）
"""
from __future__ import annotations

import logging

from config import APP
from modules.llm_agent import get_agent

logger = logging.getLogger("LumiLink.action")


class ActionEngine:
    """破冰与线下行动生成引擎。"""

    def generate(self, match_result_text: str) -> str:
        """根据匹配结果文本，生成完整的破冰行动指南（Markdown）。"""
        template = get_agent().load_prompt(APP.icebreaker_prompt_path.name)
        prompt = template.format(match_result=match_result_text)

        try:
            guide = get_agent().chat(
                prompt,
                system=(
                    "你是一位高情商校园社交顾问。请根据匹配结果，"
                    "生成温暖、具体、可执行的破冰行动指南。"
                    "输出使用 Markdown 格式，包含清晰的小标题。"
                ),
            )
            return guide or "⚠️ 未能生成破冰指南，请重试。"
        except Exception as e:
            logger.exception("破冰生成失败")
            return f"⚠️ 破冰生成失败：{e}"
