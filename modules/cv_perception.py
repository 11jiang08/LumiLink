# modules/cv_perception.py
"""
多模态感知模块 (CV Perception Layer)
核心功能：
1. ResNet18 图像场景分类 (操场/图书馆/食堂等)
2. YOLOv8 目标检测 (识别特征物品，如球拍、习题册、电脑等)
3. 多模态融合：结合用户输入的文字与图像识别结果，生成统一结构化 JSON 标签

自训练权重支持（resnet18training 分支合并）：
- 若 config.SCENE_MODEL_PATH 指向的文件存在，自动加载自训练 ResNet18 做场景分类。
- 若不存在，回退到 ImageNet 预训练 + YOLO 启发式场景判定。
"""

import logging
import os
import config

logger = logging.getLogger(__name__)

# 全局模型实例变量（延迟加载）
yolo_model = None
resnet_model = None
resnet_transforms = None
# 自训练场景类别名（英文），None 表示当前用的是 ImageNet 预训练
scene_class_names = None

# 自训练模型英文标签 → 系统中文场景名 映射
_SCENE_LABEL_CN = {
    "library_study": "图书馆/自习室",
    "canteen": "食堂/校园咖啡厅",
    "playground": "操场跑道",
    "dormitory": "宿舍",
    "classroom": "教室/自习室",
    "cafe": "校园咖啡厅",
}


def _init_models():
    """按需延迟初始化 CV 模型，避免启动卡顿"""
    global yolo_model, resnet_model, resnet_transforms, scene_class_names

    # 1. 初始化 YOLOv8
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(config.YOLO_MODEL_PATH)
            logger.info("YOLOv8 模型加载成功！")
        except Exception as e:
            logger.warning(f"YOLOv8 加载失败 (使用模拟解析保底): {e}")

    # 2. 初始化 ResNet18
    if resnet_model is None:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            from pathlib import Path

            custom_path = Path(config.SCENE_MODEL_PATH)
            if custom_path.exists():
                # 优先加载自训练场景分类权重
                checkpoint = torch.load(custom_path, map_location="cpu")
                scene_class_names = list(checkpoint.get("classes", []))
                resnet_model = models.resnet18(weights=None)
                resnet_model.fc = torch.nn.Linear(
                    resnet_model.fc.in_features, len(scene_class_names)
                )
                resnet_model.load_state_dict(checkpoint["state_dict"])
                resnet_model.eval()
                logger.info(
                    f"自训练场景权重加载成功：{custom_path}（{len(scene_class_names)} 类）"
                )
            else:
                resnet_model = models.resnet18(pretrained=True)
                resnet_model.eval()
                scene_class_names = None
                logger.info("未找到自训练权重，使用 ImageNet 预训练 ResNet18")

            resnet_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            logger.warning(f"ResNet18 加载失败 (使用模拟解析保底): {e}")


def _classify_scene_with_resnet(image_input):
    """用自训练 ResNet18 做场景分类，返回中文场景标签；不可用则返回 None。"""
    if resnet_model is None or scene_class_names is None:
        return None
    try:
        import torch
        from PIL import Image
        import numpy as np

        img = image_input
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        tensor = resnet_transforms(img).unsqueeze(0)
        with torch.no_grad():
            idx = int(resnet_model(tensor).argmax(1).item())
        if 0 <= idx < len(scene_class_names):
            en_label = scene_class_names[idx]
            return _SCENE_LABEL_CN.get(en_label, en_label)
        return None
    except Exception as e:
        logger.warning(f"ResNet18 场景分类失败，回退启发式: {e}")
        return None


def analyze_image(image_input) -> dict:
    """
    对上传的照片进行 CV 感知分析

    :param image_input: PIL Image 或 numpy array (来自 Gradio)
    :return: {"detected_scene": "操场跑道", "detected_objects": ["球拍", "水杯"], "extracted_tags": ["夜跑", "羽毛球"]}
    """
    if image_input is None:
        return {
            "detected_scene": "未知场景",
            "detected_objects": [],
            "extracted_tags": []
        }

    _init_models()

    detected_objects = []
    extracted_tags = []
    detected_scene = "校园广场"

    # --- 1. YOLOv8 物品检测 ---
    if yolo_model is not None:
        try:
            results = yolo_model(image_input)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = yolo_model.names[cls_id]
                    detected_objects.append(class_name)

                    # 映射到兴趣标签
                    if class_name in config.OBJECT_INTEREST_MAPPING:
                        tag = config.OBJECT_INTEREST_MAPPING[class_name]
                        if tag not in extracted_tags:
                            extracted_tags.append(tag)
        except Exception as e:
            logger.error(f"YOLOv8 推理异常: {e}")

    # --- 2. 场景判定 ---
    # 优先用自训练 ResNet18 做场景分类
    resnet_scene = _classify_scene_with_resnet(image_input)
    if resnet_scene:
        detected_scene = resnet_scene
    else:
        # 回退到启发式 / ResNet 场景判定
        # 如果检测到电脑/书本 -> 图书馆/自习室
        if "laptop" in detected_objects or "book" in detected_objects:
            detected_scene = "图书馆/自习室"
        # 如果检测到球类/运动器材 -> 操场
        elif "sports ball" in detected_objects or "racket" in detected_objects:
            detected_scene = "操场跑道"
        # 如果检测到杯子/餐具 -> 食堂/咖啡厅
        elif "cup" in detected_objects or "bowl" in detected_objects:
            detected_scene = "食堂/校园咖啡厅"
        else:
            detected_scene = "校园公共区域"

    # 保底：若没识别出标签，根据场景补齐默认标签
    if not extracted_tags:
        if detected_scene == "图书馆/自习室":
            extracted_tags = ["赶论文/刷题", "安静自习"]
        elif detected_scene == "操场跑道":
            extracted_tags = ["跑步/运动", "户外组队"]
        else:
            extracted_tags = ["破冰交流", "找搭子"]

    return {
        "detected_scene": detected_scene,
        "detected_objects": list(set(detected_objects)),
        "extracted_tags": extracted_tags
    }


def fuse_multimodal_inputs(text_intent: str, user_interests: list, image_input) -> dict:
    """
    多模态融合函数：将文字意图、手动兴趣与视觉感知结果融合
    """
    cv_result = analyze_image(image_input)

    # 融合兴趣标签
    final_tags = list(set(user_interests + cv_result["extracted_tags"]))

    # 确认当下场景
    final_scene = cv_result["detected_scene"]
    if final_scene == "未知场景" or final_scene == "校园公共区域":
        # 如果图片没判定出来，从文字推导
        if "跑" in text_intent or "操场" in text_intent:
            final_scene = "操场跑道"
        elif "图书馆" in text_intent or "高数" in text_intent or "赶作业" in text_intent:
            final_scene = "图书馆/自习室"
        elif "火锅" in text_intent or "食堂" in text_intent or "吃" in text_intent:
            final_scene = "食堂/餐厅"
        else:
            final_scene = "校园大门/看台"

    return {
        "final_scene": final_scene,
        "final_tags": final_tags,
        "cv_details": cv_result
    }
