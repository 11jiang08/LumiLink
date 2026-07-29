"""
用户画像与问卷结构化 (User Profile)
====================================
三大维度：
    1. 基础兴趣 (hobbies)
    2. 性格雷点 (landmines) —— 绝对不能忍受的特质
    3. 个人缺点 (weaknesses) —— 加密区，仅对 AI 可见

附加上视觉感知层提供的：
    - 当前场景 (cv_scene)
    - 识别物品 (cv_objects)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class UserProfile:
    """单个用户的完整画像。"""

    user_id: str = ""
    nickname: str = ""
    hobbies: list[str] = field(default_factory=list)
    landmines: list[str] = field(default_factory=list)       # 雷点
    weaknesses: list[str] = field(default_factory=list)      # 个人缺点（加密）
    personality: list[str] = field(default_factory=list)     # 性格特质（正向）
    cv_scene: str = ""                                        # 当前场景
    cv_objects: list[str] = field(default_factory=list)      # 识别物品

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        """对外可见版本（隐藏 weaknesses）。"""
        d = asdict(self)
        d.pop("weaknesses", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=str(data.get("user_id", "")),
            nickname=str(data.get("nickname", "")),
            hobbies=list(data.get("hobbies", [])),
            landmines=list(data.get("landmines", [])),
            weaknesses=list(data.get("weaknesses", [])),
            personality=list(data.get("personality", [])),
            cv_scene=str(data.get("cv_scene", "")),
            cv_objects=list(data.get("cv_objects", [])),
        )

    # ---------- 文本呈现（喂给 LLM） ----------
    def to_llm_text(self, reveal_weaknesses: bool = True) -> str:
        lines = [
            f"昵称：{self.nickname}",
            f"基础兴趣：{', '.join(self.hobbies) or '（无）'}",
            f"性格雷点：{', '.join(self.landmines) or '（无）'}",
        ]
        if self.personality:
            lines.append(f"性格特质：{', '.join(self.personality)}")
        if reveal_weaknesses:
            lines.append(f"个人缺点（加密区）：{', '.join(self.weaknesses) or '（无）'}")
        if self.cv_scene:
            lines.append(f"当前场景：{self.cv_scene}")
        if self.cv_objects:
            lines.append(f"识别物品/兴趣标签：{', '.join(self.cv_objects)}")
        return "\n".join(lines)


@dataclass
class MatchResult:
    """匹配引擎的输出结构。"""

    matched_user_id: str = ""
    matched_nickname: str = ""
    compatibility_score: float = 0.0          # 匹配契合度 0~1
    avoid_mine_success: float = 0.0           # 避雷成功率 0~1
    reason: str = ""                          # 匹配理由（对用户正向呈现）
    hidden_analysis: str = ""                 # 隐藏的缺点比对细节（仅调试/答辩用）

    def to_markdown(self) -> str:
        return (
            f"## 💞 匹配结果\n\n"
            f"- **匹配对象**：{self.matched_nickname}（ID: {self.matched_user_id}）\n"
            f"- **匹配契合度**：{self.compatibility_score * 100:.0f}%\n"
            f"- **避雷成功率**：{self.avoid_mine_success * 100:.0f}%\n\n"
            f"### 匹配理由\n{self.reason}\n"
        )
