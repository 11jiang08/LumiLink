# modules/__init__.py
"""
LumiLink（微光相遇） - 核心功能模块包
"""

# 1. 从多模态感知模块暴露核心函数
from .cv_perception import analyze_image, fuse_multimodal_inputs

# 2. 从 LLM 逆向匹配模块暴露核心函数
from .llm_matcher import match_user

# 3. 从线下破冰引擎模块暴露核心函数
from .action_engine import generate_icebreaker_and_guide

# 定义 __all__ 明确导出列表
__all__ = [
    "analyze_image",
    "fuse_multimodal_inputs",
    "match_user",
    "generate_icebreaker_and_guide",
]