"""
用户画像与问卷结构化 (User Profile)
====================================
三大核心维度：
    1. 基础兴趣 (hobbies)
    2. 性格雷点 (landmines) —— 绝对不能忍受的特质
    3. 个人缺点 (weaknesses) —— 加密区，仅对 AI 可见

附带视觉感知层数据：
    - 当前场景 (cv_scene)
    - 识别物品 (cv_objects)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class UserProfile:
    """单个用户的完整画像。"""

    user_id: str = ""
    nickname: str = ""
    hobbies: List[str] = field(default_factory=list)
    landmines: List[str] = field(default_factory=list)       # 绝对不接受的雷点
    weaknesses: List[str] = field(default_factory=list)      # 个人缺点（加密，仅 AI 可见）
    personality: List[str] = field(default_factory=list)     # 正向性格标签
    cv_scene: str = ""                                       # 视觉感知：场景
    cv_objects: List[str] = field(default_factory=list)      # 视觉感知：物品/兴趣标签

    # ---------- 序列化 & 反序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        """转化为完整字典格式（包含加密区）。"""
        return asdict(self)

    def to_public_dict(self) -> Dict[str, Any]:
        """对外可见版本（完全擦除加密的 weaknesses 缺点区）。"""
        d = asdict(self)
        d.pop("weaknesses", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserProfile:
        """从字典还原对象（包含自动化数据清洗）。"""
        def _clean_list(raw_list: Any) -> List[str]:
            if isinstance(raw_list, str):
                raw_list = [item.strip() for item in raw_list.split(",") if item.strip()]
            if isinstance(raw_list, list):
                return [str(item).strip() for item in raw_list if str(item).strip()]
            return []

        return cls(
            user_id=str(data.get("user_id", "")).strip(),
            nickname=str(data.get("nickname", "热心同学")).strip(),
            hobbies=_clean_list(data.get("hobbies")),
            landmines=_clean_list(data.get("landmines")),
            weaknesses=_clean_list(data.get("weaknesses")),
            personality=_clean_list(data.get("personality")),
            cv_scene=str(data.get("cv_scene", "")).strip(),
            cv_objects=_clean_list(data.get("cv_objects")),
        )

    # ---------- 辅助属性 ----------
    @property
    def has_cv_data(self) -> bool:
        """检查是否有有效的视觉感知数据。"""
        return bool(self.cv_scene or self.cv_objects)

    @property
    def is_valid(self) -> bool:
        """基础必填项校验（至少需要昵称和个人缺点才能进行逆向匹配）。"""
        return bool(self.nickname and self.weaknesses)

    # ---------- 文本呈现（喂给 LLM） ----------
    def to_llm_text(self, reveal_weaknesses: bool = True) -> str:
        """
        转换为格式化的 Prompt 文本。
        :param reveal_weaknesses: 是否露出加密缺点区。给 LLM 匹配算法喂数据时传 True，
                                  生成公开对外评价时传 False。
        """
        lines = [
            f"ID: {self.user_id or 'N/A'}",
            f"昵称：{self.nickname}",
            f"基础兴趣：{', '.join(self.hobbies) if self.hobbies else '（无）'}",
            f"性格雷点：{', '.join(self.landmines) if self.landmines else '（无）'}",
        ]
        if self.personality:
            lines.append(f"正向性格特质：{', '.join(self.personality)}")
        
        if reveal_weaknesses:
            lines.append(f"个人缺点（加密区·仅AI可见）：{', '.join(self.weaknesses) if self.weaknesses else '（无）'}")
        
        if self.cv_scene:
            lines.append(f"当前视觉场景：{self.cv_scene}")
        if self.cv_objects:
            lines.append(f"感知物品/意图标签：{', '.join(self.cv_objects)}")
            
        return "\n".join(lines)


@dataclass
class MatchResult:
    """匹配引擎的输出结构。"""

    matched_user_id: str = ""
    matched_nickname: str = ""
    compatibility_score: float = 0.0          # 匹配契合度 (0.0 - 1.0)
    avoid_mine_success: float = 0.0           # 避雷成功率 (0.0 - 1.0)
    reason: str = ""                          # 对用户呈现的高情商匹配理由
    hidden_analysis: str = ""                 # 隐藏的缺点比对逻辑（仅调试/答辩演示使用）

    def __post_init__(self):
        """确保得分在 0~1 的合理区间。"""
        if self.compatibility_score > 1.0:
            self.compatibility_score /= 100.0
        if self.avoid_mine_success > 1.0:
            self.avoid_mine_success /= 100.0

    def to_markdown(self) -> str:
        """生成渲染到 Gradio 界面的 Markdown 格式字符串。"""
        comp_pct = int(self.compatibility_score * 100)
        avoid_pct = int(self.avoid_mine_success * 100)
        
        md_text = (
            f"### 💞 匹配成功！为您精选的最佳搭子：【{self.matched_nickname}】\n\n"
            f"- **匹配契合度**：`{comp_pct}%`\n"
            f"- **安全避雷率**：`{avoid_pct}%`\n\n"
            f"#### 🌟 AI 匹配深度解析\n"
            f"> {self.reason or '你们在性格与当下场景意图中达到了天然的契合。'}\n"
        )
        
        if self.hidden_analysis:
            md_text += f"\n<details><summary>🔍 答辩/调试区：查看缺点排雷剖析（点击展开）</summary>\n\n```text\n{self.hidden_analysis}\n```\n</details>"
            
        return md_text