"""
ResNet18 校园场景分类训练脚本
================================
目标：训练一个 6 类校园场景分类模型，替换 ImageNet 预训练 + 规则映射的兜底方案，
     显著提升 cv_perception.py 在校园场景下的识别准确率。

数据集目录结构（务必按此准备）：
    data/scene_dataset/
        ├── library_study/      # 图书馆/学习
        ├── canteen/            # 食堂
        ├── playground/         # 操场
        ├── dormitory/          # 宿舍
        ├── classroom/          # 教室
        └── cafe/               # 咖啡厅/猫咖

子文件夹名 = 类别名，必须与 config.py 中的 CV.scene_labels 完全一致。
每个子文件夹内放对应场景的 .jpg / .png 图片（建议每类 80~150 张，多角度多光照）。

运行：
    python training/train_scene.py
可选参数：
    python training/train_scene.py --epochs 20 --batch-size 32 --lr 1e-3

输出：
    models/scene_resnet18.pt   （包含 state_dict + 类别标签列表）
训练完成后无需改动任何代码，cv_perception.py 会自动检测并加载该权重。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------- 路径 ----------
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data" / "scene_dataset"
MODELS_DIR: Path = ROOT / "models"
OUTPUT_PATH: Path = MODELS_DIR / "scene_resnet18.pt"

# 期望的类别顺序（与 config.CV.scene_labels 一致）
EXPECTED_CLASSES: list[str] = [
    "library_study",
    "canteen",
    "playground",
    "dormitory",
    "classroom",
    "cafe",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train")


# =========================================================
# 数据增强
# =========================================================
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """返回 (训练增强, 验证增强)。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# =========================================================
# 模型构建
# =========================================================
def build_model(num_classes: int) -> nn.Module:
    """加载 ImageNet 预训练 ResNet18，替换最后 fc 层为 num_classes 输出。"""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================================================
# 训练 / 验证一轮
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="LumiLink 校园场景分类训练")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 工作进程数")
    args = parser.parse_args()

    # ---------- 数据集 ----------
    if not DATA_DIR.exists():
        logger.error("数据集目录不存在：%s", DATA_DIR)
        logger.error(
            "请按以下结构准备图片：\n"
            "  data/scene_dataset/<类别名>/*.jpg\n"
            "类别名必须为：%s",
            EXPECTED_CLASSES,
        )
        return

    train_tf, val_tf = get_transforms()
    # 用两份 ImageFolder 实例分别套用不同增强，保证 train/val 走不同 transform
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    val_full = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    class_names: list[str] = train_full.classes
    logger.info("数据集类别顺序：%s", class_names)

    if class_names != EXPECTED_CLASSES:
        logger.warning(
            "⚠️ 类别顺序与 config.CV.scene_labels 不一致！\n"
            "  期望：%s\n  实际：%s\n"
            "  请调整子文件夹名或顺序。继续训练但不影响保存。",
            EXPECTED_CLASSES, class_names,
        )

    n_total = len(train_full)
    if n_total == 0:
        logger.error("数据集为空，请检查图片是否放到对应子文件夹内。")
        return

    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    # 同一份 indices 用于切两个 dataset，保证样本对应
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_set = Subset(train_full, indices[:n_train])
    val_set = Subset(val_full, indices[n_train:])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info("样本数：训练 %d，验证 %d（共 %d）", n_train, n_val, n_total)

    # ---------- 模型 / 损失 / 优化器 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备：%s", device)
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ---------- 训练循环 ----------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            "Epoch %02d/%02d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": class_names,
                    "val_acc": val_acc,
                },
                OUTPUT_PATH,
            )
            logger.info("✅ 验证准确率提升，已保存到 %s", OUTPUT_PATH)

    logger.info("训练完成。最佳验证准确率：%.4f", best_val_acc)
    logger.info("权重文件：%s", OUTPUT_PATH)
    logger.info("下一步：重启 app.py，cv_perception.py 会自动加载该权重。")


if __name__ == "__main__":
    main()
