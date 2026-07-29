"""
LumiLink 全局配置
=================
所有可调参数、API 配置、模型路径集中在此管理。
敏感信息（API Key）请放在 .env 文件中，切勿硬编码或提交到 git。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- 路径配置 ----------
ROOT_DIR: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = ROOT_DIR / "data"
PROMPTS_DIR: Path = ROOT_DIR / "prompts"
MODELS_DIR: Path = ROOT_DIR / "models"
ASSETS_DIR: Path = ROOT_DIR / "assets"

for _d in (DATA_DIR, PROMPTS_DIR, MODELS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- 环境变量 ----------
load_dotenv(ROOT_DIR / ".env")


class LLMConfig:
    """大语言模型调用配置（兼容 OpenAI 协议，可接 DeepSeek / GPT-4o 等）。"""

    api_key: str = os.getenv("LLM_API_KEY", "")
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))


class CVConfig:
    """视觉感知层配置。"""

    # ResNet18 场景分类标签（演示用，可替换为自训练权重）
    scene_labels: list[str] = [
        "library_study",   # 图书馆/学习
        "canteen",         # 食堂
        "playground",      # 操场
        "dormitory",       # 宿舍
        "classroom",       # 教室
        "cafe",            # 咖啡厅/猫咖
    ]
    resnet_weights: str | None = None      # None 表示用 torchvision 预训练权重
    yolo_weights: str = "yolov8n.pt"       # YOLOv8 nano，轻量快速
    yolo_conf_threshold: float = 0.35
    yolo_top_k: int = 5                    # 只取置信度最高的前 K 个物品


class AppConfig:
    """应用级配置。"""

    app_title: str = "2030 微光相遇 · Lumina Campus Link"
    app_desc: str = "基于真实自我剖析与多模态感知的校园交友系统"
    sample_users_path: Path = DATA_DIR / "sample_users.json"
    matching_prompt_path: Path = PROMPTS_DIR / "matching_prompt.txt"
    icebreaker_prompt_path: Path = PROMPTS_DIR / "icebreaker_prompt.txt"
    server_port: int = 7860
    share: bool = False


# 单例
LLM = LLMConfig()
CV = CVConfig()
APP = AppConfig()
