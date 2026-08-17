# -*- coding: utf-8 -*-
"""
Bài 1: U-Net — Vai trò của Skip Connection trong Phân đoạn ảnh (Oxford-IIIT Pet)

Mô tả:
- Tự xây dựng U-Net bằng PyTorch và huấn luyện trên tập dữ liệu Oxford-IIIT Pet.
- So sánh U-Net (có Skip Connection) với U-Net Without Skip (Ablation study).
- Đánh giá vai trò của Skip Connection trong việc phục hồi thông tin không gian,
  đặc biệt ở vùng biên (border) và các chi tiết nhỏ.

Các thành phần chính:
1. Cấu hình & Utility (Seed, Normalization, Colorize)
2. Dataset & DataLoader (Oxford-IIIT Pet, Preprocessing, Resize, Shift mask labels)
3. Kiến trúc mô hình (DoubleConv, EncoderBlock, DecoderBlock, UNet, UNetWithoutSkip)
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
IMAGE_SIZE = 128
BATCH_SIZE = 16
NUM_CLASSES = 3  # 0: pet, 1: background, 2: border
SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

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
    Dataset wrapper cho Oxford-IIIT Pet Segmentation.

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

        # Resize ảnh (Bilinear) & Mask (Nearest)
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
# 3. KIẾN TRÚC MÔ HÌNH (U-NET & U-NET WITHOUT SKIP)
# =========================================================
class DoubleConv(nn.Module):
    """Khối 2 lớp Conv 3x3 + BatchNorm + ReLU liên tiếp."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Khối Encoder: DoubleConv -> MaxPool2d (trả về cả skip feature và downsampled feature)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Khối Decoder: ConvTranspose2d -> Concat Skip Feature (nếu có) -> DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Mô hình U-Net chuẩn với Skip Connections giữa Encoder và Decoder."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(3, base_channels)                  # 32
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)  # 64
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4) # 128

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8) # 256

        # Decoder (in_channels, skip_channels, out_channels)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4) # 256 -> 128
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2) # 128 -> 64
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels)         # 64 -> 32

        # Head 1x1 Conv
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        b = self.bottleneck(p3)

        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


class UNetWithoutSkip(nn.Module):
    """Mô hình U-Net Ablation: Không sử dụng Skip Connections từ Encoder sang Decoder."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(3, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder (skip_channels = 0)
        self.dec3 = DecoderBlock(base_channels * 8, 0, base_channels * 4)
        self.dec2 = DecoderBlock(base_channels * 4, 0, base_channels * 2)
        self.dec1 = DecoderBlock(base_channels * 2, 0, base_channels)

        # Head 1x1 Conv
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, p1 = self.enc1(x)
        _, p2 = self.enc2(p1)
        _, p3 = self.enc3(p2)

        b = self.bottleneck(p3)

        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return self.out_conv(d1)


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
    """Kiểm tra kích thước đầu ra của mô hình U-Net và UNetWithoutSkip."""
    set_seed(SEED)
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    unet_test = UNet(num_classes=NUM_CLASSES)
    out_unet = unet_test(dummy_input)
    assert out_unet.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), f"Shape error in UNet: {out_unet.shape}"

    no_skip_test = UNetWithoutSkip(num_classes=NUM_CLASSES)
    out_no_skip = no_skip_test(dummy_input)
    assert out_no_skip.shape == (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE), f"Shape error in UNetWithoutSkip: {out_no_skip.shape}"

    print(f"[TEST] Shape test PASSED: Input {dummy_input.shape} -> Output {out_unet.shape}")


# =========================================================
# 6. HÀM THỰC THI CHÍNH (MAIN)
# =========================================================
def main():
    set_seed(SEED)
    print(f"[INFO] Sử dụng thiết bị: {DEVICE}")

    # 1. Shape Test
    test_shapes()

    # 2. Chuẩn bị Dữ liệu
    print("[INFO] Đang chuẩn bị dữ liệu Oxford-IIIT Pet...")
    train_dataset, val_dataset, train_loader, val_loader = prepare_dataloaders()
    print(f"[INFO] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 3. Huấn luyện U-Net (Có Skip Connection)
    print("\n" + "=" * 50)
    print(" HUẤN LUYỆN U-NET (WITH SKIP CONNECTIONS)")
    print("=" * 50)
    set_seed(SEED)
    unet = UNet(num_classes=NUM_CLASSES)
    unet, unet_history = train_model(unet, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 4. Huấn luyện U-Net Without Skip (Baseline / Ablation)
    print("\n" + "=" * 50)
    print(" HUẤN LUYỆN U-NET WITHOUT SKIP (BASELINE)")
    print("=" * 50)
    set_seed(SEED)
    no_skip = UNetWithoutSkip(num_classes=NUM_CLASSES)
    no_skip, no_skip_history = train_model(no_skip, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE)

    # 5. So sánh kết quả định lượng
    print("\n" + "=" * 50)
    print(" BẢNG SO SÁNH KẾT QUẢ ĐỊNH LƯỢNG")
    print("=" * 50)
    histories = {"U-Net": unet_history, "U-Net (No Skip)": no_skip_history}

    df_comp = pd.DataFrame([
        {
            "Model": "U-Net (With Skip)",
            "Val Loss": unet_history["val_loss"][-1],
            "Val mIoU": unet_history["val_iou"][-1]
        },
        {
            "Model": "U-Net (Without Skip)",
            "Val Loss": no_skip_history["val_loss"][-1],
            "Val mIoU": no_skip_history["val_iou"][-1]
        }
    ])
    print(df_comp.to_string(index=False))

    # 6. Trực quan hóa Biểu đồ & Kết quả dự đoán
    plot_history(histories)
    compare_predictions([unet, no_skip], val_dataset, ["U-Net (With Skip)", "U-Net (No Skip)"], count=3)


if __name__ == "__main__":
    main()