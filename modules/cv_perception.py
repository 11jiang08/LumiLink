"""
CV 感知层 (CV Perception Layer)
================================
目标：无感获取用户当下的状态与兴趣。

- ResNet18：轻量快速的场景分类（操场 / 图书馆 / 食堂 / 宿舍 / 教室 / 猫咖 …）
- YOLOv8：图文复合识别，提取画面中的关键物品（羽毛球拍、科幻小说、咖啡 …）

输出结构化 JSON：
    {
        "current_scene": "canteen",
        "detected_objects": ["hotpot", "coke"]
    }

说明：为降低高中生演示门槛，本模块默认使用 torchvision 预训练 ResNet18
      + YOLOv8n 官方 COCO 权重。后续可替换为自训练的校园场景权重。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import CV

logger = logging.getLogger("LumiLink.cv")


class CVPerceiver:
    """视觉感知统一入口。"""

    def __init__(self) -> None:
        self._resnet = None
        self._yolo = None
        self._scene_labels = CV.scene_labels
        # 物品 -> 兴趣标签 的映射（COCO 类别子集，演示用）
        self._object_to_interest: dict[str, str] = {
            "book": "阅读/小说",
            "sports ball": "球类运动",
            "baseball bat": "球类运动",
            "tennis racket": "羽毛球/网球",
            "bottle": "饮品",
            "cup": "咖啡/茶饮",
            "laptop": "数码/编程",
            "cell phone": "数码",
            "backpack": "出行",
            "handbag": "出行",
            "guitar": "音乐",
            "umbrella": "出行",
        }

    # ---------- 模型懒加载 ----------
    def _load_resnet(self):
        if self._resnet is None:
            import torch
            from torchvision import models, transforms

            logger.info("加载 ResNet18 预训练权重 ...")
            weights = (
                models.ResNet18_Weights.DEFAULT
                if CV.resnet_weights is None
                else None
            )
            model = models.resnet18(weights=weights)
            model.eval()
            self._resnet = model
            self._resnet_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            # 若未自训练，使用 ImageNet 1000 类做近似映射
            self._imagenet_labels = (
                models.ResNet18_Weights.DEFAULT.meta["categories"]
                if weights is not None
                else []
            )
        return self._resnet

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO

            logger.info("加载 YOLOv8n 权重 ...")
            self._yolo = YOLO(CV.yolo_weights)
        return self._yolo

    # ---------- 主流程 ----------
    def perceive(self, image_path: str | Path) -> dict[str, Any]:
        """对单张图片执行场景分类 + 物品检测，返回结构化结果。"""
        image_path = str(image_path)
        scene = self._classify_scene(image_path)
        objects = self._detect_objects(image_path)
        return {
            "current_scene": scene,
            "detected_objects": objects,
        }

    # ---------- ResNet18 场景分类 ----------
    def _classify_scene(self, image_path: str) -> str:
        """对当前演示版，用 ImageNet 预训练结果做规则映射到自定义场景标签。"""
        try:
            import torch

            model = self._load_resnet()
            import cv2
            import numpy as np

            bgr = cv2.imread(image_path)
            if bgr is None:
                return "未知"
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                idx = int(logits.argmax(1).item())
            label = self._imagenet_labels[idx] if self._imagenet_labels else ""

            # ImageNet 标签 -> 校园场景 的简单规则映射
            label_lower = label.lower()
            if any(k in label_lower for k in ("library", "bookshop", "book")):
                return "library_study"
            if any(k in label_lower for k in ("dining", "restaurant", "cafeteria", "tray")):
                return "canteen"
            if any(k in label_lower for k in ("stadium", "court", "gym")):
                return "playground"
            if any(k in label_lower for k in ("desk", "monitor", "computer")):
                return "classroom"
            if any(k in label_lower for k in ("bed", "dorm")):
                return "dormitory"
            if any(k in label_lower for k in ("cafe", "espresso")):
                return "cafe"
            return "未分类"
        except Exception as e:
            logger.warning("ResNet18 场景分类失败，使用兜底：%s", e)
            return "未分类"

    # ---------- YOLOv8 物品检测 ----------
    def _detect_objects(self, image_path: str) -> list[str]:
        try:
            yolo = self._load_yolo()
            results = yolo(
                image_path,
                conf=CV.yolo_conf_threshold,
                verbose=False,
            )
            names = results[0].names if results else {}
            # 取置信度 Top-K 类别（去重）
            seen: list[str] = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                order = boxes.conf.argsort(descending=True).tolist()
                for i in order:
                    cls_id = int(boxes.cls[i].item())
                    cls_name = names.get(cls_id, f"cls_{cls_id}")
                    interest = self._object_to_interest.get(cls_name, cls_name)
                    if interest not in seen:
                        seen.append(interest)
                    if len(seen) >= CV.yolo_top_k:
                        break
            return seen
        except Exception as e:
            logger.warning("YOLOv8 检测失败，返回空列表：%s", e)
            return []


# 便捷单例
_default_perceiver = CVPerceiver()


def perceive(image_path: str | Path) -> dict[str, Any]:
    """模块级快捷函数。"""
    return _default_perceiver.perceive(image_path)
