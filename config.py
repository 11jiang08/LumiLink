import os
from pathlib import Path

# 自动读取项目根目录下的 .env 文件（如果安装了 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 项目根目录路径（自动计算）
BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. 大模型 (LLM) 配置
# ==========================================
# 优先读取 OPENAI_API_KEY，其次读取 API_KEY，若都没有则默认为 None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

LLM_MODEL = "deepseek-chat"

# 检查 API Key 是否配置（用于启动时安全预警）
if not OPENAI_API_KEY:
    print("⚠️ 警告: 未检测到 OPENAI_API_KEY，请在环境变量或 .env 文件中配置，否则 LLM 匹配功能无法正常运行！")

# ==========================================
# 2. 视觉模型 (CV) 配置
# ==========================================
# YOLOv8 预训练模型路径
YOLO_MODEL_PATH = str(BASE_DIR / "yolov8n.pt")

# 自训练场景分类模型绝对路径（规避相对路径失效问题）
SCENE_MODEL_PATH = str(BASE_DIR / "models" / "scene_resnet18.pt")

# ResNet18 常用场景关键词映射字典
SCENE_MAPPING = {
    "library": "图书馆",
    "bookshop": "图书馆/书店",
    "classroom": "教室/自习室",
    "running_track": "操场跑道",
    "stadium": "体育馆/操场",
    "cafeteria": "食堂/餐厅",
    "dining": "食堂火锅/餐饮",
    "park": "校园草坪/公园",
    "coffee_shop": "校园咖啡厅"
}

# 常见物品（YOLO检测类别）映射到兴趣标签
OBJECT_INTEREST_MAPPING = {
    "book": "阅读/学习",
    "laptop": "编程/赶论文",
    "sports ball": "球类运动",
    "racket": "羽毛球/网球",
    "cell phone": "数码/游戏",
    "cup": "喝咖啡/聊天",
    "backpack": "户外/自习搭子"
}

# ResNet18 场景分类标签（必须与 data/scene_dataset/ 子文件夹名一一对应）
scene_labels: list[str] = [
    "cafe",            # 咖啡厅/猫咖
    "canteen",         # 食堂
    "classroom",       # 教室
    "dormitory",       # 宿舍
    "library_study",   # 图书馆/学习
    "playground",      # 操场
]
# 自训练场景分类权重路径。
# 若文件存在 → 自动加载自训练 ResNet18（推荐，识别准确率高）；
# 若文件不存在 → 回退到 ImageNet 预训练 + 规则关键词映射（兜底，准确率较低）。
# 训练完成后会自动生成在 models/scene_resnet18.pt，无需手动改路径。
scene_model_path: Path = BASE_DIR / "models" / "scene_resnet18.pt"
yolo_weights: str = "yolov8n.pt"       # YOLOv8 nano，轻量快速
yolo_conf_threshold: float = 0.35
yolo_top_k: int = 5                    # 只取置信度最高的前 K 个物品

# ==========================================
# 3. 全局可选项与预设配置
# ==========================================
WEAKNESS_OPTIONS = [
    "社恐/不善言辞",
    "拖延症/经常赶截止时间",
    "重度颜控/外貌协会",
    "容易焦虑/情绪化",
    "熟人疯子/生人高冷",
    "选择困难症"
]

TABOO_OPTIONS = [
    "极其讨厌别人迟到",
    "讨厌社交大吵大闹",
    "排斥过度打探隐私",
    "讨厌负能量爆棚",
    "讨厌说话不回/冷暴力"
]