import os

# ==========================================
# 1. 大模型 (LLM) 配置
# ==========================================
# 推荐使用 DeepSeek 或 OpenAI API，如果使用 OpenAI 兼容接口，修改 base_url 即可
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-a6009cd2fb3c4b838fa04a4848dd4ef1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")  # 或 https://api.openai.com/v1

LLM_MODEL = "deepseek-chat"  # 可替换为 "gpt-4o-mini" / "gpt-4o" 等

# ==========================================
# 2. 视觉模型 (CV) 配置
# ==========================================
# YOLOv8 预训练模型名称或路径 (默认使用轻量级 yolov8n.pt，首次运行会自动下载)
YOLO_MODEL_PATH = "yolov8n.pt"

# ResNet18 常用场景关键词映射字典 (用于将分类提取的通用标签转化为校园场景)
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

# ==========================================
# 3. 全局可选项与预设配置
# ==========================================
# 预设的性格缺点选项（用于 Gradio 界面多选/下拉）
WEAKNESS_OPTIONS = [
    "社恐/不善言辞", 
    "拖延症/经常赶截止时间", 
    "重度颜控/外貌协会", 
    "容易焦虑/情绪化", 
    "熟人疯子/生人高冷", 
    "选择困难症"
]

# 预设的性格雷点选项（绝对不能接受的特质）
TABOO_OPTIONS = [
    "极其讨厌别人迟到", 
    "讨厌社交大吵大闹", 
    "排斥过度打探隐私", 
    "讨厌负能量爆棚", 
    "讨厌说话不回/冷暴力"
]