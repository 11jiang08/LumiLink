"""
多模态感知模块 (CV Perception Layer)
核心功能：
1. ResNet 场景分类 (自训练权重优先，未找到则回退 ImageNet 预训练 + 启发式)
2. YOLOv8 目标检测 (识别特征物品，映射用户兴趣标签)
3. 多模态融合：结合文字意图、用户手动标签与视觉感知生成最终结构化 JSON
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import numpy as np

import config

logger = logging.getLogger(__name__)

# 自训练模型英文标签 → 系统中文场景名 映射
_SCENE_LABEL_CN = {
    "library_study": "图书馆/自习室",
    "canteen": "食堂/校园咖啡厅",
    "playground": "操场跑道",
    "dormitory": "宿舍",
    "classroom": "教室/自习室",
    "cafe": "校园咖啡厅",
}


class CVPerceptionEngine:
    """CV 感知引擎（单例模式，延迟加载模型）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CVPerceptionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._yolo_model = None
        self._resnet_model = None
        self._resnet_transform = None
        self._scene_class_names: Optional[List[str]] = None
        self._device = self._get_device()
        self._initialized = True

    def _get_device(self) -> str:
        """获取当前可用的加速设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _init_yolo(self):
        """懒加载 YOLOv8 目标检测模型"""
        if self._yolo_model is not None:
            return

        try:
            from ultralytics import YOLO
            model_path = getattr(config, "YOLO_MODEL_PATH", "yolov8n.pt")
            self._yolo_model = YOLO(model_path)
            logger.info("YOLOv8 模型加载成功！[Device: %s]", self._device)
        except Exception as e:
            logger.error("YOLOv8 模型加载失败: %s", e)
            self._yolo_model = None

    def _init_resnet(self):
        """懒加载 ResNet 场景分类模型"""
        if self._resnet_model is not None:
            return

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._resnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            custom_path = Path(getattr(config, "SCENE_MODEL_PATH", "weights/scene_resnet.pt"))
            
            if custom_path.exists():
                logger.info("尝试加载自训练场景模型: %s", custom_path)
                checkpoint = torch.load(custom_path, map_location=self._device)
                
                # 兼容不同 Checkpoint 保存格式
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    self._scene_class_names = checkpoint.get("classes", list(_SCENE_LABEL_CN.keys()))
                else:
                    state_dict = checkpoint
                    self._scene_class_names = list(_SCENE_LABEL_CN.keys())

                # 自动适应 ResNet18 或 ResNet50 结构
                num_classes = len(self._scene_class_names)
                try:
                    # 优先尝试 ResNet18
                    model = models.resnet18(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                    model.load_state_dict(state_dict)
                except Exception:
                    # 尝试带 Dropout 的 ResNet50 结构
                    model = models.resnet50(weights=None)
                    model.fc = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.3),
                        torch.nn.Linear(model.fc.in_features, num_classes)
                    )
                    model.load_state_dict(state_dict)

                model.to(self._device)
                model.eval()
                self._resnet_model = model
                logger.info("自训练 ResNet 场景权重加载成功！（类别数：%d）", num_classes)
            else:
                # ImageNet 预训练回退模式
                logger.warning("未找到自训练权重，加载 ImageNet 预训练 ResNet18 模型...")
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                model.to(self._device)
                model.eval()
                self._resnet_model = model
                self._scene_class_names = None

        except Exception as e:
            logger.error("ResNet 加载失败 (降级到纯启发式模式): %s", e)
            self._resnet_model = None

    def _preprocess_image(self, image_input: Any) -> Optional[Image.Image]:
        """将各种类型的图像输入转换为 RGB 格式的 PIL.Image"""
        if image_input is None:
            return None
        try:
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                return Image.fromarray(image_input).convert("RGB")
            elif isinstance(image_input, (str, Path)) and os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
        except Exception as e:
            logger.error("图像预处理失败: %s", e)
        return None

    def classify_scene(self, img: Image.Image) -> Optional[str]:
        """使用 ResNet 进行场景分类，返回中文场景名称"""
        self._init_resnet()
        if self._resnet_model is None or self._resnet_transform is None:
            return None

        try:
            import torch
            tensor = self._resnet_transform(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                outputs = self._resnet_model(tensor)
                idx = int(outputs.argmax(1).item())

            if self._scene_class_names and 0 <= idx < len(self._scene_class_names):
                raw_label = self._scene_class_names[idx]
                return _SCENE_LABEL_CN.get(raw_label, raw_label)
        except Exception as e:
            logger.warning("ResNet 场景推理异常: %s", e)
            
        return None

    def detect_objects(self, image_input: Any, conf_threshold: float = 0.25) -> List[str]:
        """使用 YOLOv8 进行目标检测，返回去重后的物体名称列表"""
        self._init_yolo()
        if self._yolo_model is None:
            return []

        detected_objects = []
        try:
            results = self._yolo_model(image_input, conf=conf_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self._yolo_model.names[cls_id]
                    detected_objects.append(class_name)
        except Exception as e:
            logger.error("YOLOv8 推理失败: %s", e)

        return list(set(detected_objects))


# 单例句柄
_engine = CVPerceptionEngine()


def analyze_image(image_input: Any) -> Dict[str, Any]:
    """
    单张图像感知分析主接口

    :param image_input: PIL Image / Numpy Array / 图像路径
    :return: {
        "detected_scene": "操场跑道",
        "detected_objects": ["sports ball", "bottle"],
        "extracted_tags": ["球类运动", "饮品"]
    }
    """
    img = _engine._preprocess_image(image_input)
    if img is None:
        return {
            "detected_scene": "未知场景",
            "detected_objects": [],
            "extracted_tags": []
        }

    # 1. YOLOv8 目标检测
    detected_objects = _engine.detect_objects(img)

    # 2. 映射检测到的物体 -> 用户兴趣标签
    extracted_tags = []
    object_mapping = getattr(config, "OBJECT_INTEREST_MAPPING", {
        "book": "阅读/学习",
        "sports ball": "球类运动",
        "tennis racket": "网球/羽毛球",
        "laptop": "数码/编程",
        "cup": "咖啡/特饮",
        "guitar": "音乐/乐器"
    })

    for obj in detected_objects:
        if obj in object_mapping:
            tag = object_mapping[obj]
            if tag not in extracted_tags:
                extracted_tags.append(tag)

    # 3. 场景判定（自训练 ResNet 优先，若无则使用启发式）
    detected_scene = _engine.classify_scene(img)

    if not detected_scene:
        # 启发式兜底逻辑
        if "laptop" in detected_objects or "book" in detected_objects:
            detected_scene = "图书馆/自习室"
        elif "sports ball" in detected_objects or "tennis racket" in detected_objects:
            detected_scene = "操场跑道"
        elif "cup" in detected_objects or "bowl" in detected_objects:
            detected_scene = "食堂/校园咖啡厅"
        else:
            detected_scene = "校园公共区域"

    # 4. 保底标签补齐
    if not extracted_tags:
        if detected_scene == "图书馆/自习室":
            extracted_tags = ["安静自习", "搭子刷题"]
        elif detected_scene == "操场跑道":
            extracted_tags = ["跑步/运动", "户外约战"]
        else:
            extracted_tags = ["破冰交流", "找搭子"]

    return {
        "detected_scene": detected_scene,
        "detected_objects": detected_objects,
        "extracted_tags": extracted_tags
    }


def fuse_multimodal_inputs(text_intent: str, user_interests: List[str], image_input: Any) -> Dict[str, Any]:
    """
    多模态融合接口：结合用户文本意图、手动选择的兴趣与 CV 感知结果
    """
    cv_result = analyze_image(image_input)

    # 融合并去重所有兴趣标签
    user_interests = user_interests or []
    final_tags = list(set(user_interests + cv_result["extracted_tags"]))

    # 确认当下最终场景
    final_scene = cv_result["detected_scene"]
    
    # 若图片识别结果不明确，由文本意图推导场景
    if final_scene in ["未知场景", "校园公共区域"] and text_intent:
        text = text_intent.lower()
        if any(k in text for k in ["跑", "操场", "健身", "球"]):
            final_scene = "操场跑道"
        elif any(k in text for k in ["图书馆", "自习", "高数", "论文", "学习"]):
            final_scene = "图书馆/自习室"
        elif any(k in text for k in ["吃", "食堂", "火锅", "咖啡", "奶茶"]):
            final_scene = "食堂/餐厅"
        else:
            final_scene = "校园公共区域"

    return {
        "final_scene": final_scene,
        "final_tags": final_tags,
        "cv_details": cv_result
    }