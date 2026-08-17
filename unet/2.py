# -*- coding: utf-8 -*-
"""
Bài 2: DeepLabV3 — Atrous Convolution & ASPP trong Phân đoạn ảnh (Oxford-IIIT Pet)

Mô tả:
- Sử dụng ResNet18 pretrained (đến layer3, output stride 16) làm backbone.
- Tự xây dựng Atrous Spatial Pyramid Pooling (ASPP) với các tỉ lệ dilation khác nhau (1, 3, 6).
- Xây dựng mô hình DeepLabV3 và so sánh với mô hình Baseline (dùng Normal Convolution Head).
- Đánh giá vai trò của Atrous Convolution và ASPP trong việc mở rộng receptive field
  và bắt thông tin đa tỷ lệ (multi-scale context).

Các thành phần chính:
1. Cấu hình & Utility (Seed, Normalization, Colorize)
2. Dataset & DataLoader (Oxford-IIIT Pet, Preprocessing 224x224, Shift mask labels)
3. Backbone & Kiến trúc mô hình (ResNet18 Backbone, ASPP, DeepLabV3, ResNetSegmentationBaseline)
4. Huấn luyện & Đánh giá (CrossEntropyLoss, Mean IoU via Confusion Matrix)
5. Trực quan hóa & So sánh kết quả
"""

import os
import random
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# =========================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ SIÊU THAM SỐ
# =========================================================
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
IN_KAGGLE = Path("/kaggle/working").exists()

if IN_COLAB:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        DATA_ROOT = "/content/drive/MyDrive/datasets/OxfordIIITPet"
    except (NotImplementedError, RuntimeError):
        print("[INFO] Không thể mount Google Drive. Lưu tại /content/data.")
        DATA_ROOT = "/content/data/OxfordIIITPet"
elif IN_KAGGLE:
    DATA_ROOT = "/kaggle/working/datasets/OxfordIIITPet"
else:
    DATA_ROOT = "./data/OxfordIIITPet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 3  # 0: pet, 1: background, 2: border
SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

BACKBONE_OUT_CHANNELS = 256
ASPP_CHANNELS = 128
HEAD_CHANNELS = 128

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = ["pet", "background", "border"]
MASK_COLORS = np.array([
    [230, 80, 80],    # pet (red)
    [70, 120, 230],   # background (blue)
    [245, 210, 70],   # border (yellow)
], dtype=np.uint8)


def set_seed(seed: int = SEED) -> None:
    """Cố định seed để đảm bảo tính tái lập của thử nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Khôi phục ảnh đã chuẩn hóa về dải [0, 1] để hiển thị."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1)


def colorize_mask(mask: torch.Tensor) -> np.ndarray:
    """Chuyển mask nhãn số nguyên thành ảnh màu RGB tương ứng."""
    return MASK_COLORS[mask.cpu().numpy()]


# =========================================================
# 2. DATASET VÀ PREPROCESSING
# =========================================================
class PetSegmentationDataset(Dataset):
    """
    Dataset wrapper cho Oxford-IIIT Pet Segmentation (ResNet 224x224).

    Trả về:
        image: Tensor [3, H, W] đã chuẩn hóa ImageNet
        mask : Tensor [H, W] chứa nhãn nguyên {0, 1, 2}
    """

    def __init__(self, subset, image_size: int = IMAGE_SIZE):
        self.subset = subset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, mask = self.subset[index]

        # Resize ảnh (Bilinear) & Mask (Nearest) về 224x224
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        # Chuyển ảnh PIL thành Tensor [0, 1] và chuẩn hóa
        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        # Shift nhãn mask từ {1, 2, 3} thành {0, 1, 2}
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) - 1

        return image, mask


def prepare_dataloaders():
    """Tải dataset Oxford-IIIT Pet và khởi tạo DataLoader."""
    base_dataset = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        target_types="segmentation",
        download=True,
    )

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    split_generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=split_generator
    )

    train_dataset = PetSegmentationDataset(train_subset, IMAGE_SIZE)
    val_dataset = PetSegmentationDataset(val_subset, IMAGE_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataset, val_dataset, train_loader, val_loader


# =========================================================
# 3. KIẾN TRÚC MÔ HÌNH (RESNET18 BACKBONE, ASPP, DEEPLABV3 & BASELINE)
# =========================================================
def build_resnet18_backbone() -> nn.Module:
    """Xây dựng ResNet18 backbone dừng ở layer3 (output stride = 16, 256 channels)."""
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )
    return backbone


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP):
    Gồm 4 nhánh song song:
    - Nhánh 1: Conv 1x1
    - Nhánh 2: Atrous Conv 3x3, dilation = 1
    - Nhánh 3: Atrous Conv 3x3, dilation = 3
    - Nhánh 4: Atrous Conv 3x3, dilation = 6
    Kết quả từ 4 nhánh được concatenate và nén qua Conv 1x1.
    """

    def __init__(self, in_channels: int = BACKBONE_OUT_CHANNELS, out_channels: int = ASPP_CHANNELS):
        super().__init__()
        # Nhánh 1: Conv 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Nhánh 2: Atrous Conv 3x3, dilation = 1 (padding = 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Nhánh 3: Atrous Conv 3x3, dilation = 3 (padding = 3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Nhánh 4: Atrous Conv 3x3, dilation = 6 (padding = 6)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection head (1x1 Conv)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.project(out)


class DeepLabV3(nn.Module):
    """
    Mô hình DeepLabV3:
    Backbone (ResNet18 up to layer3) -> ASPP -> 1x1 Classifier -> Bilinear Upsample.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, aspp_channels: int = ASPP_CHANNELS):
        super().__init__()
        self.backbone = build_resnet18_backbone()
        self.aspp = ASPP(in_channels=BACKBONE_OUT_CHANNELS, out_channels=aspp_channels)
        self.classifier = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feat = self.backbone(x)
        feat = self.aspp(feat)
        logits = self.classifier(feat)
        out = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return out


class ResNetSegmentationBaseline(nn.Module):
    """
    Mô hình Baseline (Ablation study):
    Cùng Backbone (ResNet18 up to layer3) nhưng dùng Normal Conv Head (dilation = 1).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, head_channels: int = HEAD_CHANNELS):
        super().__init__()
        self.backbone = build_resnet18_backbone()
        self.head = nn.Sequential(
            nn.Conv2d(BACKBONE_OUT_CHANNELS, head_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feat = self.backbone(x)
        logits = self.head(feat)
        out = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return out


# =========================================================
# 4. THÀNH PHẦN HUẤN LUYỆN VÀ ĐÁNH GIÁ (TRAINING UTILITIES)
# =========================================================
@torch.no_grad()
def update_confusion_stats(preds: torch.Tensor, targets: torch.Tensor,
                           num_classes: int, intersection_sum: torch.Tensor,
                           union_sum: torch.Tensor) -> None:
    """Cộng dồn intersection và union của từng lớp trong epoch."""
    for class_id in range(num_classes):
        pred_class = (preds == class_id)
        target_class = (targets == class_id)
        intersection_sum[class_id] += (pred_class & target_class).sum().item()
        union_sum[class_id] += (pred_class | target_class).sum().item()


def compute_mean_iou(intersection_sum: torch.Tensor, union_sum: torch.Tensor) -> float:
    """Tính mIoU trung bình cho các lớp có union > 0."""
    valid = union_sum > 0
    if not valid.any():
        return float("nan")
    return (intersection_sum[valid] / union_sum[valid]).mean().item()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer: torch.optim.Optimizer = None):
    """Chạy 1 epoch huấn luyện hoặc validation."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_items = 0
    intersection_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    union_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False, desc="Training" if is_training else "Validation"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            if not is_training:
                preds = logits.argmax(dim=1)
                update_confusion_stats(preds, masks, NUM_CLASSES, intersection_sum, union_sum)

    mean_loss = total_loss / total_items
    mean_iou = None if is_training else compute_mean_iou(intersection_sum, union_sum)
    return mean_loss, mean_iou


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE):
    """Hàm huấn luyện mô hình toàn vẹn qua các epochs."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_iou:.4f} | Time: {elapsed:.1f}s"
        )

    return model, history


# =========================================================
# 5. TRỰC QUAN HÓA & SO SÁNH
# =========================================================
def show_dataset_samples(dataset, count: int = 3) -> None:
    """Hiển thị một số mẫu ảnh đầu vào và ground truth mask."""
    fig, axes = plt.subplots(count, 2, figsize=(7, 3 * count))
    for row in range(count):
        image, mask = dataset[row]
        axes[row, 0].imshow(denormalize(image).permute(1, 2, 0))
        axes[row, 0].set_title("Input Image")
        axes[row, 1].imshow(colorize_mask(mask))
        axes[row, 1].set_title("Ground Truth Mask")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_history(histories: dict) -> None:
    """Vẽ biểu đồ so sánh Loss và mIoU giữa các mô hình."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], marker="o", label=f"{label} Train")
        axes[0].plot(epochs, history["val_loss"], marker="s", linestyle="--", label=f"{label} Val")
        axes[1].plot(epochs, history["val_iou"], marker="o", label=label)

    axes[0].set(title="Loss Curve", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[1].set(title="Validation mIoU", xlabel="Epoch", ylabel="mIoU")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_mask(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Dự đoán mask cho một bức ảnh đơn lẻ."""
    model.eval()
    logits = model(image.unsqueeze(0).to(DEVICE))
    return logits.argmax(dim=1).squeeze(0).cpu()


def compare_predictions(models: list, dataset, model_names: list, count: int = 3) -> None:
    """So sánh kết quả dự đoán của các mô hình với Ground Truth."""
    columns = 2 + len(models)
    fig, axes = plt.subplots(count, columns, figsize=(4 * columns, 3.5 * count))

    for row in range(count):
        image, target = dataset[row]
        axes[row, 0].imshow(denormalize(image).permute(1, 2, 0))
        axes[row, 0].set_title("Input Image")
        axes[row, 1].imshow(colorize_mask(target))
        axes[row, 1].set_title("Ground Truth")

        for col, (model, name) in enumerate(zip(models, model_names), start=2):
            pred = predict_mask(model, image)
            axes[row, col].imshow(colorize_mask(pred))
            axes[row, col].set_title(name)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def test_shapes() -> None:
    """Kiểm tra kích thước đầu ra của Backbone, DeepLabV3 và ResNetSegmentationBaseline."""
    set_seed(SEED)
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    # 1. Test Backbone
    backbone_check = build_resnet18_backbone().eval()
    feat_check = backbone_check(dummy_input)
    assert feat_check.shape == (2, BACKBONE_OUT_CHANNELS, 14, 14), f"Shape error in Backbone: {feat_check.shape}"

    # 2. Test DeepLabV3
    deeplab_test = DeepLabV3(num_classes=NUM_CLASSES)
    out_deeplab = deeplab_test(dummy_input)
    assert out_deeplab.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), f"Shape error in DeepLabV3: {out_deeplab.shape}"

    # 3. Test Baseline
    baseline_test = ResNetSegmentationBaseline(num_classes=NUM_CLASSES)
    out_baseline = baseline_test(dummy_input)
    assert out_baseline.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), f"Shape error in Baseline: {out_baseline.shape}"

    print(f"[TEST] Shape test PASSED: Input {dummy_input.shape} -> Output {out_deeplab.shape}")


# =========================================================
# 6. HÀM THỰC THI CHÍNH (MAIN)
# =========================================================
def main():
    set_seed(SEED)
    print(f"[INFO] Sử dụng thiết bị: {DEVICE}")

    # 1. Shape Test
    test_shapes()

    # 2. Chuẩn bị Dữ liệu
    print("[INFO] Đang chuẩn bị dữ liệu Oxford-IIIT Pet (224x224)...")
    train_dataset, val_dataset, train_loader, val_loader = prepare_dataloaders()
    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Huấn luyện ResNet18 + ASPP (DeepLabV3)
    print("\n" + "=" * 50)
    print(" HUẤN LUYỆN DEEPLABV3 (RESNET18 + ASPP)")
    print("=" * 50)
    set_seed(SEED)
    deeplab = DeepLabV3(num_classes=NUM_CLASSES)
    deeplab, deeplab_history = train_model(deeplab, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 4. Huấn luyện ResNet18 + Normal Conv (Baseline)
    print("\n" + "=" * 50)
    print(" HUẤN LUYỆN BASELINE (RESNET18 + NORMAL CONV)")
    print("=" * 50)
    set_seed(SEED)
    baseline = ResNetSegmentationBaseline(num_classes=NUM_CLASSES)
    baseline, baseline_history = train_model(baseline, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 5. So sánh kết quả định lượng
    print("\n" + "=" * 50)
    print(" BẢNG SO SÁNH KẾT QUẢ ĐỊNH LƯỢNG")
    print("=" * 50)
    histories = {
        "ResNet18 + ASPP (DeepLabV3)": deeplab_history,
        "ResNet18 + Normal Conv (Baseline)": baseline_history
    }

    df_comp = pd.DataFrame([
        {
            "Model": "ResNet18 + ASPP (DeepLabV3)",
            "Val Loss": deeplab_history["val_loss"][-1],
            "Val mIoU": deeplab_history["val_iou"][-1]
        },
        {
            "Model": "ResNet18 + Normal Conv (Baseline)",
            "Val Loss": baseline_history["val_loss"][-1],
            "Val mIoU": baseline_history["val_iou"][-1]
        }
    ])
    print(df_comp.to_string(index=False))

    # 6. Trực quan hóa Biểu đồ & Kết quả dự đoán
    plot_history(histories)
    compare_predictions([deeplab, baseline], val_dataset, ["DeepLabV3 (ASPP)", "Baseline (Normal Conv)"], count=3)


if __name__ == "__main__":
    main()
