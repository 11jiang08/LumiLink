"""
校园场景分类训练脚本（增强版）
================================
针对小数据集（每类 50-150 张）优化的高准确率训练方案：

核心改进（相比初版）：
1. 模型升级：ResNet18 → ResNet50（更强特征提取，25M 参数）
2. 分阶段训练：先冻结 backbone 只训分类头（10 epoch），再解冻全量微调
   —— 避免预训练权重被小数据集破坏
3. 更强数据增强：+ RandomAffine + RandomErasing（模拟遮挡）
4. CosineAnnealingLR：平滑退火，替换粗暴的 StepLR
5. 标签平滑：label_smoothing=0.1，提升泛化
6. 混合精度训练：GPU 自动启用 AMP，加速 + 轻微正则
7. 早停：patience=8，过拟合时自动停止
8. TTA 验证：测试时增强，验证准确率再提 1-3%

数据集目录结构：
    data/scene_dataset/
        ├── library_study/      # 图书馆/学习
        ├── canteen/            # 食堂
        ├── playground/         # 操场
        ├── dormitory/          # 宿舍
        ├── classroom/          # 教室
        └── cafe/               # 咖啡厅/猫咖

运行：
    python training/train_scene.py
可选参数：
    python training/train_scene.py --epochs 50 --batch-size 32 --lr 1e-3

输出：
    models/scene_resnet18.pt   （文件名保留兼容，实际是 ResNet50 权重）
训练完成后无需改动任何代码，cv_perception.py 会自动检测并加载。
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------- 路径 ----------
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data" / "scene_dataset"
MODELS_DIR: Path = ROOT / "models"
OUTPUT_PATH: Path = MODELS_DIR / "scene_resnet18.pt"  # 文件名保留兼容

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
# 数据增强（强化版）
# =========================================================
def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """返回 (训练增强, 验证增强)。"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # 新增：随机仿射变换（平移+缩放+剪切），增强视角泛化
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # 新增：随机擦除，模拟遮挡，显著提升泛化
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# =========================================================
# 模型构建（ResNet50 + 分阶段冻结）
# =========================================================
def build_model(num_classes: int) -> nn.Module:
    """加载 ImageNet 预训练 ResNet50，替换 fc 层 + 加 Dropout 防过拟合。"""
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    # fc 层加 Dropout，防止小数据集过拟合
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """冻结除分类头外的所有层（阶段一：只训 fc）。"""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("阶段一：backbone 冻结，可训练参数 %d / %d (%.1f%%)",
                trainable, total, 100 * trainable / total)


def unfreeze_all(model: nn.Module):
    """解冻所有层（阶段二：全量微调）。"""
    for param in model.parameters():
        param.requires_grad = True
    logger.info("阶段二：全部解冻，全量微调")


# =========================================================
# 训练 / 验证（支持混合精度）
# =========================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    use_amp = scaler is not None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


@torch.no_grad()
def evaluate_tta(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    indices: list[int],
    device: torch.device,
) -> float:
    """TTA（测试时增强）：对每张图做 3 种变体取平均预测，提升验证准确率。"""
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    flip_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    correct, total = 0, 0
    for idx in indices:
        img, label = dataset[idx]
        # 原图 + 水平翻转，取 softmax 平均
        imgs = torch.stack([base_tf(img), flip_tf(img)]).to(device)
        logits = model(imgs)
        avg_prob = torch.softmax(logits, dim=1).mean(dim=0)
        pred = avg_prob.argmax().item()
        correct += (pred == label)
        total += 1
    return correct / max(total, 1)


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="LumiLink 校园场景分类训练（增强版）")
    parser.add_argument("--epochs", type=int, default=40, help="总训练轮数（含冻结阶段）")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="阶段一（冻结）轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="阶段一学习率")
    parser.add_argument("--lr-finetune", type=float, default=1e-4, help="阶段二微调学习率")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 工作进程数")
    parser.add_argument("--patience", type=int, default=8, help="早停耐心值（连续 N 轮无提升则停）")
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
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_set = Subset(train_full, indices[:n_train])
    val_set = Subset(val_full, indices[n_train:])
    val_indices = indices[n_train:]

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info("样本数：训练 %d，验证 %d（共 %d）", n_train, n_val, n_total)

    # ---------- 模型 / 损失 / 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备：%s", device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("启用混合精度训练（AMP）")

    model = build_model(num_classes=len(class_names)).to(device)
    # 标签平滑，提升泛化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---------- 阶段一：冻结 backbone，只训 fc ----------
    logger.info("=" * 60)
    logger.info("阶段一：冻结 backbone，只训练分类头（%d epoch）", args.freeze_epochs)
    logger.info("=" * 60)
    freeze_backbone(model)
    optimizer_stage1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler_stage1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_stage1, T_max=args.freeze_epochs
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.freeze_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_stage1, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler_stage1.step()
        logger.info(
            "[阶段一] Epoch %02d/%02d (%.0fs) | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
            epoch, args.freeze_epochs, time.time() - t0, train_loss, train_acc, val_loss, val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(), "classes": class_names, "val_acc": val_acc},
                OUTPUT_PATH,
            )
            logger.info("✅ 验证准确率提升，已保存到 %s", OUTPUT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

    # ---------- 阶段二：解冻全量微调 ----------
    finetune_epochs = args.epochs - args.freeze_epochs
    if finetune_epochs > 0:
        logger.info("=" * 60)
        logger.info("阶段二：解冻全部层，全量微调（%d epoch，lr=%g）", finetune_epochs, args.lr_finetune)
        logger.info("=" * 60)
        unfreeze_all(model)
        optimizer_stage2 = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
        scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_stage2, T_max=finetune_epochs
        )

        for epoch in range(1, finetune_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer_stage2, device, scaler
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler_stage2.step()
            logger.info(
                "[阶段二] Epoch %02d/%02d (%.0fs) | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f",
                epoch, finetune_epochs, time.time() - t0, train_loss, train_acc, val_loss, val_acc,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"state_dict": model.state_dict(), "classes": class_names, "val_acc": val_acc},
                    OUTPUT_PATH,
                )
                logger.info("✅ 验证准确率提升，已保存到 %s", OUTPUT_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("⏹️ 连续 %d 轮无提升，触发早停", args.patience)
                    break

    # ---------- TTA 最终验证 ----------
    logger.info("=" * 60)
    logger.info("最终评估：使用 TTA（测试时增强）重新验证")
    logger.info("=" * 60)
    tta_acc = evaluate_tta(model, val_full, val_indices, device)
    logger.info("TTA 验证准确率：%.4f（普通验证最佳：%.4f）", tta_acc, best_val_acc)

    # TTA 结果更好就覆盖保存
    if tta_acc > best_val_acc:
        best_val_acc = tta_acc
        torch.save(
            {"state_dict": model.state_dict(), "classes": class_names, "val_acc": tta_acc},
            OUTPUT_PATH,
        )
        logger.info("✅ TTA 准确率更高，已更新保存")

    logger.info("=" * 60)
    logger.info("训练完成。最佳准确率：%.4f", best_val_acc)
    logger.info("权重文件：%s", OUTPUT_PATH)
    logger.info("下一步：重启 app.py，cv_perception.py 会自动加载该权重。")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
