
import math
import os
import random
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms, box_iou
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

# =========================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ SIÊU THAM SỐ
# =========================================================
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
IN_KAGGLE = Path("/kaggle/working").exists()

if IN_COLAB:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        DATA_ROOT = Path("/content/drive/MyDrive/datasets/road_damage_pcm")
    except (ImportError, RuntimeError):
        DATA_ROOT = Path("/content/data/road_damage_pcm")
elif IN_KAGGLE:
    DATA_ROOT = Path("/kaggle/working/datasets/road_damage_pcm")
else:
    DATA_ROOT = Path("./data/road_damage_pcm")

IMAGE_SIZE = 640
BATCH_SIZE = 8
NUM_CLASSES = 3
CLASS_NAMES = ["Pothole", "Crack", "Manhole"]
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 0
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTO_DOWNLOAD_DATASET = True

# YOLOv8 sử dụng 16 bins DFL và dự đoán tại 3 scale stride (8, 16, 32)
REG_MAX = 16
STRIDES = (8, 16, 32)
BOX_GAIN, CLS_GAIN, DFL_GAIN = 7.5, 0.5, 1.5

assert IMAGE_SIZE % 32 == 0, "IMAGE_SIZE phải chia hết cho 32."


def set_seed(seed: int = SEED) -> None:
    """Cố định seed để đảm bảo tính tái lập của thử nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================================================
# 2. TẢI VÀ CHUẨN HÓA DATASET TỰ ĐỘNG
# =========================================================
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
)
DATASET_ARCHIVE = DATA_ROOT.parent / "road_damage_pcm.zip"


def prepare_road_damage_dataset(data_root: Path = DATA_ROOT) -> None:
    """Tải, chia 80/20 và đưa dataset về cấu trúc YOLO train/valid."""
    ready_file = data_root / ".ready"
    if ready_file.exists():
        print(f"Dataset đã sẵn sàng tại: {data_root}")
        return

    DATASET_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_ARCHIVE.exists() or not zipfile.is_zipfile(DATASET_ARCHIVE):
        print("Đang tải Road Damage Dataset (~195 MB)...")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_ARCHIVE)
        except Exception as err:
            print("Không thể tải dataset:", err)
            return

    image_prefix = "data/images/"
    label_prefix = "data/labels-YOLO/"
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with zipfile.ZipFile(DATASET_ARCHIVE) as archive:
        names = archive.namelist()
        labels = set(name for name in names if name.startswith(label_prefix))
        pairs = []
        for image_name in names:
            if not image_name.startswith(image_prefix) or Path(image_name).suffix.lower() not in image_exts:
                continue
            label_name = f"{label_prefix}{Path(image_name).stem}.txt"
            if label_name in labels:
                pairs.append((image_name, label_name))

        random.Random(SEED).shuffle(pairs)
        split_index = int(0.8 * len(pairs))
        for index, (image_name, label_name) in enumerate(tqdm(pairs, desc="Chuẩn hóa dataset")):
            split = "train" if index < split_index else "valid"
            for member, folder in ((image_name, "images"), (label_name, "labels")):
                destination = data_root / split / folder / Path(member).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

    data_yaml = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": CLASS_NAMES,
    }
    with (data_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False, allow_unicode=True)
    ready_file.write_text(str(len(pairs)), encoding="utf-8")
    print(f"Hoàn tất: {split_index} train | {len(pairs) - split_index} validation")


if AUTO_DOWNLOAD_DATASET:
    try:
        prepare_road_damage_dataset()
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print("Không thể tải dataset:", error)
        print("Shape test và loss test vẫn chạy được mà không cần dataset.")


# =========================================================
# 3. DATASET VÀ DATALOADER
# =========================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YOLODetectionDataset(Dataset):
    """Đọc ảnh và label YOLO normalized: class_id, x_center, y_center, width, height."""

    def __init__(self, data_root: Path, split: str, image_size: int):
        self.image_dir = Path(data_root) / split / "images"
        self.label_dir = Path(data_root) / split / "labels"
        self.image_size = image_size
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy {self.image_dir}.")
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = TF.to_tensor(TF.resize(image, [self.image_size, self.image_size], antialias=True))

        labels, boxes = [], []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x, y, width, height = line.split()
                labels.append(int(class_id))
                boxes.append([float(x), float(y), float(width), float(height)])

        target = {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        }
        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_dataloaders():
    train_dataset = YOLODetectionDataset(DATA_ROOT, "train", IMAGE_SIZE)
    val_dataset = YOLODetectionDataset(DATA_ROOT, "valid", IMAGE_SIZE)
    options = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    return train_dataset, val_dataset, train_loader, val_loader


try:
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()
    DATA_AVAILABLE = True
    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
except FileNotFoundError as error:
    DATA_AVAILABLE = False
    train_dataset = val_dataset = train_loader = val_loader = None
    print(error)
    print("Shape test và loss test vẫn chạy được mà không cần dataset.")


# =========================================================
# 4. TRỰC QUAN HÓA DATASET
# =========================================================
def draw_boxes(image, boxes, labels, scores=None, class_names=CLASS_NAMES):
    """Vẽ bounding box và nhãn class lên ảnh PIL/Tensor."""
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image.cpu())
    else:
        image = image.copy()

    draw = ImageDraw.Draw(image)
    colors = ["#E65050", "#4678E6", "#F5D246", "#50E678", "#E650E6"]

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = colors[int(label) % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        class_name = class_names[int(label)] if class_names and int(label) < len(class_names) else str(label)
        caption = f"{class_name}"
        if scores is not None:
            caption += f" {scores[i]:.2f}"

        draw.text((x1 + 4, y1 + 4), caption, fill=color)

    return image


def visualize_dataset(dataset, count=3):
    """Trực quan hóa một số mẫu ảnh và ground-truth box từ dataset."""
    if dataset is None or len(dataset) == 0:
        print("Dataset trống hoặc không có sẵn.")
        return
    count = min(count, len(dataset))
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5))
    if count == 1:
        axes = [axes]

    for i in range(count):
        image, target = dataset[i]
        gt_boxes = xywhn_to_xyxy(target["boxes"], image.shape[-1])
        annotated = draw_boxes(image, gt_boxes, target["labels"])
        axes[i].imshow(annotated)
        axes[i].set_title(f"Sample {i + 1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 5. KIẾN TRÚC MÔ HÌNH YOLOV8
# =========================================================

def autopad(kernel_size, padding=None, dilation=1):
    """Tính padding để giữ spatial size khi stride=1."""
    if padding is None:
        if isinstance(kernel_size, int):
            padding = (kernel_size - 1) * dilation // 2
        else:
            padding = [(k - 1) * dilation // 2 for k in kernel_size]
    return padding


class Conv(nn.Module):
    """Khối Conv-BN-SiLU tiêu chuẩn trong YOLOv8."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, dilation=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            nn.SiLU(inplace=True)
            if activation is True
            else (
                activation
                if isinstance(activation, nn.Module)
                else nn.Identity()
            )
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Khối Bottleneck tiêu chuẩn của YOLOv8 với residual shortcut."""

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=3, stride=1)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Khối Cross Stage Partial Network với 2 Convs (C2f) của YOLOv8."""

    def __init__(self, in_channels, out_channels, repeats=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.cv2 = Conv((2 + repeats) * self.hidden_channels, out_channels, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(
                self.hidden_channels,
                self.hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
            )
            for _ in range(repeats)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for layer in self.m:
            y.append(layer(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLOv8Backbone(nn.Module):
    """Backbone YOLOv8n trích xuất bộ ba đặc trưng P3, P4, P5."""

    def __init__(self):
        super().__init__()
        # Stem -> P1/2 (16 channels)
        self.stem = Conv(3, 16, 3, 2)

        # Stage 1 -> P2/4 (32 channels)
        self.stage1_conv = Conv(16, 32, 3, 2)
        self.stage1_c2f = C2f(32, 32, repeats=1, shortcut=True)

        # Stage 2 -> P3/8 (64 channels)
        self.stage2_conv = Conv(32, 64, 3, 2)
        self.stage2_c2f = C2f(64, 64, repeats=2, shortcut=True)

        # Stage 3 -> P4/16 (128 channels)
        self.stage3_conv = Conv(64, 128, 3, 2)
        self.stage3_c2f = C2f(128, 128, repeats=2, shortcut=True)

        # Stage 4 -> P5/32 (256 channels)
        self.stage4_conv = Conv(128, 256, 3, 2)
        self.stage4_c2f = C2f(256, 256, repeats=1, shortcut=True)
        self.sppf = SPPF(256, 256, kernel_size=5)

        self.out_channels = (64, 128, 256)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1_c2f(self.stage1_conv(x))

        p3 = self.stage2_c2f(self.stage2_conv(x))   # Stride 8,  Channels: 64
        p4 = self.stage3_c2f(self.stage3_conv(p3))  # Stride 16, Channels: 128
        p5 = self.sppf(
            self.stage4_c2f(self.stage4_conv(p4))
        )  # Stride 32, Channels: 256

        return p3, p4, p5


class YOLOv8Neck(nn.Module):
    """Multi-scale Feature Fusion Neck: Kết hợp FPN (Top-down) và PAN (Bottom-up)."""

    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels  # 64, 128, 256

        # Top-down pathway (FPN)
        self.c2f_up4 = C2f(c5 + c4, c4, repeats=1, shortcut=False)
        self.c2f_up3 = C2f(c4 + c3, c3, repeats=1, shortcut=False)

        # Bottom-up pathway (PAN)
        self.conv_down3 = Conv(c3, c3, 3, 2)
        self.c2f_down4 = C2f(c3 + c4, c4, repeats=1, shortcut=False)
        self.conv_down4 = Conv(c4, c4, 3, 2)
        self.c2f_down5 = C2f(c4 + c5, c5, repeats=1, shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, features):
        p3_in, p4_in, p5_in = features

        # Top-down FPN
        p5_up = F.interpolate(p5_in, scale_factor=2, mode="nearest")
        p4_up = self.c2f_up4(torch.cat([p5_up, p4_in], dim=1))

        p4_up_up = F.interpolate(p4_up, scale_factor=2, mode="nearest")
        p3_out = self.c2f_up3(torch.cat([p4_up_up, p3_in], dim=1))

        # Bottom-up PAN
        p3_down = self.conv_down3(p3_out)
        p4_out = self.c2f_down4(torch.cat([p3_down, p4_up], dim=1))

        p4_down = self.conv_down4(p4_out)
        p5_out = self.c2f_down5(torch.cat([p4_down, p5_in], dim=1))

        return p3_out, p4_out, p5_out


class DetectionHead(nn.Module):
    """Anchor-free Decoupled Detection Head (tách riêng bbox DFL branch và classification branch)."""

    def __init__(self, channels, num_classes, reg_max=REG_MAX):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.box_branches = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, 4 * reg_max, 1)
            )
            for c in channels
        ])

        self.cls_branches = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3, 1), Conv(c, c, 3, 1), nn.Conv2d(c, num_classes, 1)
            )
            for c in channels
        ])

    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            bbox_logits = self.box_branches[i](feat)
            class_logits = self.cls_branches[i](feat)
            outputs.append((bbox_logits, class_logits))
        return outputs


class YOLOv8(nn.Module):
    """Mô hình YOLOv8 hoàn chỉnh kết hợp Backbone, Neck và Detection Head."""

    def __init__(self, num_classes=NUM_CLASSES, reg_max=REG_MAX):
        super().__init__()
        self.backbone = YOLOv8Backbone()
        self.neck = YOLOv8Neck(self.backbone.out_channels)
        self.head = DetectionHead(self.neck.out_channels, num_classes, reg_max)

    def forward(self, x):
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)
        return self.head(neck_feats)


# =========================================================
# 6. BOX UTILITIES VÀ DFL DECODING
# =========================================================
def xywh_to_xyxy(boxes):
    """Chuyển box center xywh sang corner xyxy."""
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat([center - size / 2, center + size / 2], dim=-1)


def xywhn_to_xyxy(boxes, image_size):
    """Chuyển box normalized xywh sang pixel xyxy."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    scale = torch.tensor([image_size, image_size, image_size, image_size], device=boxes.device)
    boxes_xywh = boxes * scale
    return xywh_to_xyxy(boxes_xywh)


def make_anchor_points(outputs, strides=STRIDES, offset=0.5):
    """Tạo grid anchor points cho 3 scale dự đoán."""
    points, stride_values = [], []
    for (_, class_logits), stride in zip(outputs, strides):
        _, _, height, width = class_logits.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=class_logits.device, dtype=class_logits.dtype) + offset,
            torch.arange(width, device=class_logits.device, dtype=class_logits.dtype) + offset,
            indexing="ij",
        )
        points.append(torch.stack([x, y], dim=-1).reshape(-1, 2))
        stride_values.append(torch.full((height * width, 1), stride, device=x.device, dtype=x.dtype))
    return torch.cat(points), torch.cat(stride_values)


def flatten_outputs(outputs):
    """Duỗi logits từ 3 scale về dạng 2D tensor tensor [Batch, Num_Points, Channels]."""
    bbox_all, class_all = [], []
    for bbox_logits, class_logits in outputs:
        batch_size = bbox_logits.shape[0]
        bbox_all.append(bbox_logits.view(batch_size, 4 * REG_MAX, -1).permute(0, 2, 1))
        class_all.append(class_logits.view(batch_size, NUM_CLASSES, -1).permute(0, 2, 1))
    return torch.cat(bbox_all, dim=1), torch.cat(class_all, dim=1)


def dist_to_box(distances, anchor_points):
    """Chuyển khoảng cách 4 hướng (left, top, right, bottom) sang tọa độ xyxy."""
    left_top, right_bottom = distances.chunk(2, dim=-1)
    return torch.cat([anchor_points - left_top, anchor_points + right_bottom], dim=-1)


def decode_dfl(bbox_logits, anchor_points, stride_tensor, reg_max=REG_MAX):
    """Giải mã 4 phân phối DFL logits thành tọa độ box xyxy theo đơn vị pixel."""
    batch_size, num_points, _ = bbox_logits.shape
    distribution = bbox_logits.view(batch_size, num_points, 4, reg_max).softmax(dim=-1)
    projection = torch.arange(reg_max, device=bbox_logits.device, dtype=bbox_logits.dtype)
    distances = distribution.matmul(projection)
    boxes_grid = dist_to_box(distances, anchor_points.unsqueeze(0))
    return boxes_grid * stride_tensor.unsqueeze(0)


def bbox_to_dist(anchor_points, boxes_grid, reg_max=REG_MAX):
    """Chuyển GT box theo đơn vị feature grid sang khoảng cách dfl target."""
    left_top, right_bottom = boxes_grid.chunk(2, dim=-1)
    distances = torch.cat([anchor_points - left_top, right_bottom - anchor_points], dim=-1)
    return distances.clamp(0, reg_max - 1 - 0.01)


# =========================================================
# 7. IOU VÀ TASK-ALIGNED ASSIGNER (TAL)
# =========================================================
def aligned_ciou(box1, box2, eps=1e-7):
    """Tính Complete IoU (CIoU) cho hai tensor box xyxy đã aligned."""
    inter_lt = torch.maximum(box1[..., :2], box2[..., :2])
    inter_rb = torch.minimum(box1[..., 2:], box2[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    intersection = inter_wh[..., 0] * inter_wh[..., 1]

    wh1 = (box1[..., 2:] - box1[..., :2]).clamp(min=eps)
    wh2 = (box2[..., 2:] - box2[..., :2]).clamp(min=eps)
    union = wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - intersection
    iou = intersection / (union + eps)

    center1 = (box1[..., :2] + box1[..., 2:]) / 2
    center2 = (box2[..., :2] + box2[..., 2:]) / 2
    center_distance = ((center1 - center2) ** 2).sum(dim=-1)
    enclosing_lt = torch.minimum(box1[..., :2], box2[..., :2])
    enclosing_rb = torch.maximum(box1[..., 2:], box2[..., 2:])
    diagonal = ((enclosing_rb - enclosing_lt) ** 2).sum(dim=-1) + eps

    v = (4 / math.pi ** 2) * (
        torch.atan(wh2[..., 0] / wh2[..., 1]) - torch.atan(wh1[..., 0] / wh1[..., 1])
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - center_distance / diagonal - alpha * v


def pairwise_ciou(boxes1, boxes2):
    return aligned_ciou(boxes1[:, None, :], boxes2[None, :, :])


class TaskAlignedAssigner:
    """Task-Aligned Assigner (TAL) phân bổ động mẫu positive dựa trên alignment metric t = s^alpha * IoU^beta."""

    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(self, class_prob, pred_boxes, anchor_pixels, gt_labels, gt_boxes):
        batch_size, num_points, num_classes = class_prob.shape
        target_boxes = torch.zeros_like(pred_boxes)
        target_scores = torch.zeros_like(class_prob)
        foreground = torch.zeros((batch_size, num_points), dtype=torch.bool, device=class_prob.device)

        for batch_index in range(batch_size):
            labels = gt_labels[batch_index]
            boxes = gt_boxes[batch_index]
            if labels.numel() == 0:
                continue

            left_top = anchor_pixels[None] - boxes[:, None, :2]
            right_bottom = boxes[:, None, 2:] - anchor_pixels[None]
            inside = torch.cat([left_top, right_bottom], dim=-1).amin(dim=-1) > 1e-9

            ious = pairwise_ciou(boxes, pred_boxes[batch_index]).clamp(min=0)
            scores_for_gt = class_prob[batch_index, :, labels].T
            alignment = scores_for_gt.pow(self.alpha) * ious.pow(self.beta)
            alignment = alignment * inside

            selected = torch.zeros_like(inside)
            k = min(self.topk, num_points)
            top_values, top_indices = alignment.topk(k, dim=1)
            selected.scatter_(1, top_indices, top_values > 0)

            selected_ious = torch.where(selected, ious, torch.full_like(ious, -1))
            best_iou, best_gt = selected_ious.max(dim=0)
            positive = best_iou >= 0
            if not positive.any():
                continue

            point_indices = positive.nonzero(as_tuple=False).squeeze(1)
            matched_gt = best_gt[positive]
            foreground[batch_index, point_indices] = True
            target_boxes[batch_index, point_indices] = boxes[matched_gt]

            for gt_index in range(len(boxes)):
                mask = positive & (best_gt == gt_index)
                if not mask.any():
                    continue
                metric = alignment[gt_index, mask]
                quality = metric / (metric.max() + 1e-9) * ious[gt_index, mask].max()
                target_scores[batch_index, mask, labels[gt_index]] = quality

        return target_boxes, target_scores, foreground


class YOLOv8DetectionLoss:
    """Tổng hợp Loss YOLOv8: CIoU Box Loss + Classification BCE Loss + Distribution Focal Loss (DFL)."""

    def __init__(self, num_classes=NUM_CLASSES, reg_max=REG_MAX):
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)

    def _prepare_targets(self, targets, image_size, device):
        labels_batch, boxes_batch = [], []
        scale = torch.tensor([image_size, image_size, image_size, image_size], device=device)
        for target in targets:
            labels_batch.append(target["labels"].to(device))
            boxes_xywh = target["boxes"].to(device) * scale
            boxes_batch.append(xywh_to_xyxy(boxes_xywh))
        return labels_batch, boxes_batch

    @staticmethod
    def _distribution_focal_loss(logits, target):
        left = target.long()
        right = left + 1
        weight_left = right - target
        weight_right = 1 - weight_left
        flat_logits = logits.reshape(-1, logits.shape[-1])
        loss_left = F.cross_entropy(flat_logits, left.reshape(-1), reduction="none").view_as(left)
        loss_right = F.cross_entropy(flat_logits, right.reshape(-1), reduction="none").view_as(right)
        return (loss_left * weight_left + loss_right * weight_right).mean(dim=-1)

    def __call__(self, outputs, targets, image_size=IMAGE_SIZE):
        bbox_logits, class_logits = flatten_outputs(outputs)
        anchor_points, stride_tensor = make_anchor_points(outputs)
        pred_boxes = decode_dfl(bbox_logits, anchor_points, stride_tensor, self.reg_max)
        labels_batch, boxes_batch = self._prepare_targets(targets, image_size, bbox_logits.device)

        target_boxes, target_scores, foreground = self.assigner(
            class_logits.detach().sigmoid(), pred_boxes.detach(),
            anchor_points * stride_tensor, labels_batch, boxes_batch,
        )
        target_score_sum = target_scores.sum().clamp(min=1.0)

        cls_loss = F.binary_cross_entropy_with_logits(
            class_logits, target_scores, reduction="sum"
        ) / target_score_sum

        if foreground.any():
            positive_weights = target_scores.sum(dim=-1)[foreground]
            ciou = aligned_ciou(pred_boxes[foreground], target_boxes[foreground])
            box_loss = ((1 - ciou) * positive_weights).sum() / target_score_sum

            target_boxes_grid = target_boxes / stride_tensor.unsqueeze(0)
            target_dist = bbox_to_dist(anchor_points.unsqueeze(0), target_boxes_grid, self.reg_max)
            positive_logits = bbox_logits[foreground].view(-1, 4, self.reg_max)
            dfl_each = self._distribution_focal_loss(positive_logits, target_dist[foreground])
            dfl_loss = (dfl_each * positive_weights).sum() / target_score_sum
        else:
            box_loss = bbox_logits.sum() * 0.0
            dfl_loss = bbox_logits.sum() * 0.0

        total = BOX_GAIN * box_loss + CLS_GAIN * cls_loss + DFL_GAIN * dfl_loss
        components = {
            "loss": total.detach(),
            "box_loss": box_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "dfl_loss": dfl_loss.detach(),
            "foreground": foreground.sum().detach(),
        }
        return total, components


# =========================================================
# 8. KIỂM TRA SMOKE TEST (SHAPE & LOSS)
# =========================================================
def run_smoke_test():
    """Kiểm tra shape tensor và tính hữu hạn của loss/backward."""
    print("--- Running Smoke Test ---")
    set_seed(SEED)
    model = YOLOv8(NUM_CLASSES, REG_MAX)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)

    model.eval()
    with torch.no_grad():
        backbone_features = model.backbone(x)
        neck_features = model.neck(backbone_features)
        outputs = model.head(neck_features)

    expected_hw = [IMAGE_SIZE // stride for stride in STRIDES]
    expected_channels = model.backbone.out_channels

    for index, (backbone_feature, neck_feature, prediction, hw, channels) in enumerate(
        zip(backbone_features, neck_features, outputs, expected_hw, expected_channels), start=3
    ):
        bbox_logits, class_logits = prediction
        assert backbone_feature.shape == (2, channels, hw, hw), f"Backbone P{index} sai shape"
        assert neck_feature.shape == (2, channels, hw, hw), f"Neck P{index} sai shape"
        assert bbox_logits.shape == (2, 4 * REG_MAX, hw, hw), f"BBox branch P{index} sai shape"
        assert class_logits.shape == (2, NUM_CLASSES, hw, hw), f"Classification branch P{index} sai shape"
        print(f"P{index}: feature={tuple(neck_feature.shape)} | bbox={tuple(bbox_logits.shape)} | cls={tuple(class_logits.shape)}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Full forward: OK | Parameters: {parameter_count / 1e6:.2f}M")

    # Smoke Test Loss & Backward
    smoke_model = YOLOv8(NUM_CLASSES, REG_MAX).to(DEVICE).train()
    smoke_images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    smoke_targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.50, 0.50, 0.35, 0.30]])},
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.35, 0.40, 0.25, 0.20]])},
    ]
    smoke_outputs = smoke_model(smoke_images)
    smoke_criterion = YOLOv8DetectionLoss(NUM_CLASSES, REG_MAX)
    smoke_loss, smoke_parts = smoke_criterion(smoke_outputs, smoke_targets, image_size=IMAGE_SIZE)
    assert torch.isfinite(smoke_loss), f"Loss không hữu hạn: {smoke_loss}"
    assert smoke_parts["foreground"].item() > 0, "Task Assigner chưa tạo positive point."
    smoke_loss.backward()
    has_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in smoke_model.parameters()
    )
    assert has_gradient, "Backward chưa tạo gradient hữu hạn."
    print("Loss components:", {key: float(value) for key, value in smoke_parts.items()})
    print("Smoke Test: Passed Successfuly!\n")
    del smoke_model, smoke_images, smoke_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 9. TRAINING LOOP VÀ ĐÁNH GIÁ (EVALUATION / NMS)
# =========================================================
def move_targets_to_device(targets, device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    totals = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}
    total_items = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            targets = move_targets_to_device(targets, DEVICE)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss, parts = criterion(outputs, targets, image_size=images.shape[-1])
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_items += batch_size
            for key in totals:
                totals[key] += float(parts[key]) * batch_size

    return {key: value / max(total_items, 1) for key, value in totals.items()}


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    model = model.to(DEVICE)
    criterion = YOLOv8DetectionLoss(NUM_CLASSES, REG_MAX)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_stats = run_epoch(model, train_loader, criterion, optimizer)
        val_stats = run_epoch(model, val_loader, criterion)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train {train_stats['loss']:.4f} | val {val_stats['loss']:.4f} | "
            f"box {val_stats['box_loss']:.4f} | cls {val_stats['cls_loss']:.4f} | "
            f"dfl {val_stats['dfl_loss']:.4f} | {time.time() - start:.1f}s"
        )
    return model, pd.DataFrame(history)


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="s", label="validation")
    for name in ("box_loss", "cls_loss", "dfl_loss"):
        axes[1].plot(history["epoch"], history[f"val_{name}"], marker="o", label=name)
    axes[0].set(title="Total loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation loss components", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    plt.tight_layout()
    plt.show()


# INFERENCE & NMS
@torch.no_grad()
def postprocess(outputs, confidence_threshold=0.25, iou_threshold=0.7, max_detections=300):
    bbox_logits, class_logits = flatten_outputs(outputs)
    anchor_points, stride_tensor = make_anchor_points(outputs)
    boxes = decode_dfl(bbox_logits, anchor_points, stride_tensor, REG_MAX)
    probabilities = class_logits.sigmoid()
    scores, labels = probabilities.max(dim=-1)
    results = []

    for image_boxes, image_scores, image_labels in zip(boxes, scores, labels):
        keep = image_scores >= confidence_threshold
        image_boxes = image_boxes[keep]
        image_scores = image_scores[keep]
        image_labels = image_labels[keep]
        if image_boxes.numel() == 0:
            results.append({
                "boxes": image_boxes.reshape(0, 4),
                "scores": image_scores,
                "labels": image_labels,
            })
            continue
        keep_indices = batched_nms(image_boxes, image_scores, image_labels, iou_threshold)
        keep_indices = keep_indices[:max_detections]
        results.append({
            "boxes": image_boxes[keep_indices],
            "scores": image_scores[keep_indices],
            "labels": image_labels[keep_indices],
        })
    return results


@torch.no_grad()
def predict_batch(model, images, confidence_threshold=0.25, iou_threshold=0.7):
    model.eval()
    outputs = model(images.to(DEVICE))
    return postprocess(outputs, confidence_threshold, iou_threshold)


# EVALUATION (AP & mAP)
def integrate_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def evaluate_records(records, num_classes=NUM_CLASSES):
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap_table = np.full((num_classes, len(iou_thresholds)), np.nan)
    precision_50, recall_50 = [], []

    for class_id in range(num_classes):
        gt_by_image = {}
        predictions = []
        total_gt = 0
        for image_id, prediction, target in records:
            gt_boxes = target["boxes"][target["labels"] == class_id]
            gt_by_image[image_id] = gt_boxes
            total_gt += len(gt_boxes)
            mask = prediction["labels"] == class_id
            for box, score in zip(prediction["boxes"][mask], prediction["scores"][mask]):
                predictions.append((float(score), image_id, box))

        if total_gt == 0:
            continue
        predictions.sort(key=lambda item: item[0], reverse=True)

        for threshold_index, threshold in enumerate(iou_thresholds):
            matched = {
                image_id: torch.zeros(len(boxes), dtype=torch.bool)
                for image_id, boxes in gt_by_image.items()
            }
            true_positive, false_positive = [], []
            for _, image_id, box in predictions:
                gt_boxes = gt_by_image[image_id]
                if len(gt_boxes) == 0:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
                    continue
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_index = ious.max(dim=0)
                if best_iou >= threshold and not matched[image_id][best_index]:
                    matched[image_id][best_index] = True
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)

            if predictions:
                tp = np.cumsum(true_positive)
                fp = np.cumsum(false_positive)
                recall_curve = tp / max(total_gt, 1)
                precision_curve = tp / np.maximum(tp + fp, 1e-9)
                ap_table[class_id, threshold_index] = integrate_ap(recall_curve, precision_curve)
                if threshold_index == 0:
                    precision_50.append(float(precision_curve[-1]))
                    recall_50.append(float(recall_curve[-1]))
            else:
                ap_table[class_id, threshold_index] = 0.0
                if threshold_index == 0:
                    precision_50.append(0.0)
                    recall_50.append(0.0)

    return {
        "Precision": float(np.mean(precision_50)) if precision_50 else float("nan"),
        "Recall": float(np.mean(recall_50)) if recall_50 else float("nan"),
        "mAP@0.5": float(np.nanmean(ap_table[:, 0])),
        "mAP@0.5:0.95": float(np.nanmean(ap_table)),
    }


@torch.no_grad()
def evaluate_model(model, loader, confidence_threshold=0.001, iou_threshold=0.7):
    model.eval()
    records = []
    image_id = 0
    for images, targets in tqdm(loader, leave=False):
        predictions = predict_batch(model, images, confidence_threshold, iou_threshold)
        for prediction, target in zip(predictions, targets):
            target_boxes = xywhn_to_xyxy(target["boxes"], images.shape[-1]).cpu()
            target_cpu = {"boxes": target_boxes, "labels": target["labels"].cpu()}
            prediction_cpu = {key: value.cpu() for key, value in prediction.items()}
            records.append((image_id, prediction_cpu, target_cpu))
            image_id += 1
    return pd.DataFrame([evaluate_records(records)])


# =========================================================
# 10. TRỰC QUAN HÓA PREDICTIONS
# =========================================================
@torch.no_grad()
def visualize_predictions(model, dataset, count=3, confidence_threshold=0.25):
    if dataset is None or len(dataset) == 0:
        print("Dataset không có sẵn để trực quan hóa dự đoán.")
        return
    count = min(count, len(dataset))
    fig, axes = plt.subplots(count, 3, figsize=(14, 4.5 * count), squeeze=False)

    for row in range(count):
        image, target = dataset[row]
        prediction = predict_batch(
            model, image.unsqueeze(0), confidence_threshold=confidence_threshold
        )[0]
        gt_boxes = xywhn_to_xyxy(target["boxes"], image.shape[-1])

        axes[row, 0].imshow(TF.to_pil_image(image))
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(draw_boxes(image, gt_boxes, target["labels"]))
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(draw_boxes(
            image, prediction["boxes"].cpu(), prediction["labels"].cpu(),
            prediction["scores"].cpu(),
        ))
        axes[row, 2].set_title("Prediction")
        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# 11. CÂU HỎI LÝ THUYẾT VÀ KẾT LUẬN
# =========================================================
"""
CÂU HỎI LÝ THUYẾT & PHÂN TÍCH:

1. Vai trò của Backbone, Multi-scale Feature Fusion (Neck) và Detection Head trong YOLOv8:
   - Backbone (Darknet-like): Trích xuất đặc trưng hình ảnh qua nhiều cấp độ downsampling (stride 8, 16, 32).
     + P3 (Low-level): Độ phân giải cao (80x80), receptive field nhỏ, phù hợp trích xuất chi tiết kết cấu và biên thể hiện các vật thể nhỏ.
     + P4 (Mid-level): Kích thước trung bình (40x40), cân bằng thông tin chi tiết và ngữ nghĩa.
     + P5 (High-level): Độ phân giải thấp (20x20), receptive field lớn và biểu diễn ngữ nghĩa trừu tượng tốt cho các vật thể kích thước lớn.
   - Multi-scale Feature Fusion (Neck - FPN + PAN):
     + Feature Pyramid Network (FPN): Truyền ngữ nghĩa giàu có từ các tầng sâu (P5 -> P4 -> P3) từ trên xuống qua Upsampling.
     + Path Aggregation Network (PAN): Truyền thông tin định vị chính xác từ các tầng nông (P3 -> P4 -> P5) từ dưới lên qua Downsampling.
     + Sự kết hợp FPN + PAN và các khối C2f giúp hòa trộn thông tin ngữ nghĩa và thông tin không gian hiệu quả cho mọi scale.
   - Detection Head (Decoupled Head & DFL):
     + YOLOv8 tách biệt 2 nhánh dự đoán: Nhánh BBox DFL (dự đoán khoảng cách 4 hướng theo phân phối xác suất 16 bins) và Nhánh Classification (dự đoán xác suất các lớp).
     + Việc tách nhánh giúp giảm bớt xung đột bài toán giữa phát hiện vị trí (regression) và phân loại lớp (classification).

2. Vì sao dự đoán trên P3, P4, P5 giúp phát hiện đối tượng kích thước khác nhau:
   - Mặc định ảnh vào 640x640:
     + P3 (stride 8): Kích thước feature map 80x80, mỗi cell quản lý vùng ảnh 8x8 pixel => Lý tưởng phát hiện vật thể nhỏ (Small objects).
     + P4 (stride 16): Kích thước feature map 40x40, mỗi cell quản lý vùng ảnh 16x16 pixel => Lý tưởng phát hiện vật thể vừa (Medium objects).
     + P5 (stride 32): Kích thước feature map 20x20, mỗi cell quản lý vùng ảnh 32x32 pixel => Lý tưởng phát hiện vật thể lớn (Large objects).

3. Điểm khác biệt chính giữa cơ chế Anchor-free của YOLOv8 và Anchor-based của các phiên bản YOLO trước:
   - Cơ chế Anchor-based (YOLOv3, YOLOv4, YOLOv5):
     + Yêu cầu thiết lập trước tập hợp các Anchor box cố định (kích thước/tỷ lệ Aspect ratio) thông qua K-Means Clustering trên dataset.
     + Tốn nhiều siêu tham số, nhạy cảm với dataset mới và độ phức tạp khi tính IoU matching lớn.
   - Cơ chế Anchor-free của YOLOv8:
     + Dự đoán trực tiếp từ vị trí tâm grid point (anchor point) đến 4 cạnh của bounding box (left, top, right, bottom).
     + Loại bỏ hoàn toàn sự phụ thuộc vào predefined anchor boxes, giúp mô hình tổng quát hóa tốt hơn và giảm thiểu lượng hyperparameter cần tinh chỉnh.
     + Kết hợp với Distribution Focal Loss (DFL), khoảng cách được dự đoán dưới dạng một phân phối xác suất liên tục qua các bins phân đoạn thay vì dự đoán một giá trị đơn lẻ, giúp tăng độ chính xác định vị biên vật thể mờ/che khuất.
"""

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # 1. Chạy Smoke Test kiểm tra mô hình & loss
    run_smoke_test()

    # 2. Nếu Dataset sẵn sàng, thực hiện Huấn luyện, Đánh giá và Visualizing
    if DATA_AVAILABLE:
        print("Dataset sẵn sàng. Bắt đầu huấn luyện mô hình YOLOv8...")
        set_seed(SEED)
        model = YOLOv8(NUM_CLASSES, REG_MAX)
        model, history = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)

        # Hiển thị lịch sử huấn luyện
        print("\n--- Lịch sử huấn luyện (Training History) ---")
        print(history.to_string(index=False))
        plot_history(history)

        # Đánh giá mô hình trên tập validation
        print("\n--- Đánh giá mAP trên tập Validation ---")
        metrics = evaluate_model(model, val_loader)
        print(metrics.to_string(index=False))

        # Trực quan hóa kết quả dự đoán
        print("\n--- Trực quan hóa dự đoán ---")
        visualize_predictions(model, val_dataset, count=3)
    else:
        print("Dataset chưa sẵn sàng. Đã hoàn tất các kiểm thử Smoke Test.")
