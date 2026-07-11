
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm


DATA_ROOT   = "/content/data/GARBAGE CLASSIFICATION"   # Colab path
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, num_classes + 1)

    def forward(self, regions):
        features = self.feature_extractor(regions)           # (N, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)          # (N, 512)
        logits   = self.classifier(features)                 # (N, num_classes+1)
        return logits



def get_region_proposals(image_np, max_proposals=100):
  
    _, regions = selectivesearch.selective_search(
        image_np, scale=500, sigma=0.9, min_size=10
    )
    proposals, seen = [], set()
    for region in regions:
        x, y, w, h = region['rect']
        if w < 20 or h < 20:
            continue
        if (x, y, w, h) in seen:
            continue
        seen.add((x, y, w, h))
        proposals.append((x, y, w, h))
        if len(proposals) >= max_proposals:
            break
    return proposals



def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = max(0,      int((cx - w/2) * img_w))
    y1 = max(0,      int((cy - h/2) * img_h))
    x2 = min(img_w,  int((cx + w/2) * img_w))
    y2 = min(img_h,  int((cy + h/2) * img_h))
    return x1, y1, x2, y2


def compute_iou(box1, box2):

    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class GarbageRCNNDataset:
    def __init__(self, split="train", transform=None,
                 max_proposals=100, iou_pos=0.5, iou_neg=0.3):
        self.transform     = transform or TRANSFORM
        self.max_proposals = max_proposals
        self.iou_pos       = iou_pos
        self.iou_neg       = iou_neg

        img_dir        = os.path.join(DATA_ROOT, split, "images")
        self.label_dir = os.path.join(DATA_ROOT, split, "labels")
        self.img_paths = (sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
                        + sorted(glob.glob(os.path.join(img_dir, "*.jpeg"))))
        print(f"[{split}] Tìm thấy {len(self.img_paths)} ảnh")

    def _load_gt(self, label_path, img_w, img_h):
        boxes, labels = [], []
        if not os.path.exists(label_path):
            return boxes, labels
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x1, y1, x2, y2 = yolo_to_xyxy(*map(float, parts[1:5]), img_w, img_h)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)          # 0 = background
        return boxes, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_pil  = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        stem       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, stem + ".txt")
        gt_boxes, gt_labels = self._load_gt(label_path, img_w, img_h)

        proposals = get_region_proposals(np.array(img_pil), self.max_proposals)

        crops, labels = [], []
        for (x, y, w, h) in proposals:
            prop_box = [x, y, x+w, y+h]
            best_iou, best_label = 0, 0
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                iou = compute_iou(prop_box, gt_box)
                if iou > best_iou:
                    best_iou, best_label = iou, gt_label

            if best_iou >= self.iou_pos:
                label = best_label
            elif best_iou <= self.iou_neg:
                label = 0
            else:
                continue

            region = img_pil.crop((x, y, x+w, y+h))
            crops.append(self.transform(region))
            labels.append(label)

        if not crops:
            return torch.zeros(1, 3, 224, 224), torch.tensor([0])

        return torch.stack(crops), torch.tensor(labels)



def train_rcnn(model, num_epochs=3, lr=1e-4, max_proposals=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset   = GarbageRCNNDataset(split="train", transform=train_transform,
                                   max_proposals=max_proposals)
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(range(len(dataset)),
                    desc=f"Epoch {epoch+1}/{num_epochs}", unit="img")

        for idx in pbar:
            crops, labels = dataset[idx]
            if crops.shape[0] == 0:
                continue

            crops  = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(crops)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            total_loss    += loss.item()

            pbar.set_postfix({
                "loss": f"{total_loss / (idx + 1):.4f}",
                "acc" : f"{total_correct / max(total_samples, 1):.3f}"
            })

        print(f"✅ Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {total_loss/len(dataset):.4f} | "
              f"Acc: {total_correct/max(total_samples,1):.3f}\n")

    return model



def rcnn_inference(model, image_path, device, threshold=0.5, max_proposals=200):
    model.eval()
    model.to(device)

    img_pil  = Image.open(image_path).convert("RGB")
    img_np   = np.array(img_pil)
    proposals = get_region_proposals(img_np, max_proposals=max_proposals)

    # Crop + preprocess từng proposal
    crops = []
    for (x, y, w, h) in proposals:
        region = img_pil.crop((x, y, x+w, y+h))
        crops.append(TRANSFORM(region))

    if not crops:
        print("Không tìm thấy proposals!")
        return []

    # Batch forward
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            batch  = torch.stack(crops[i:i+64]).to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    probs      = torch.softmax(all_logits, dim=1)
    scores, pred_cls = probs.max(dim=1)

    results = []
    for i, (x, y, w, h) in enumerate(proposals):
        cls_id = pred_cls[i].item()
        score  = scores[i].item()
        if cls_id == 0 or score < threshold:
            continue
        results.append({
            "box"  : (x, y, w, h),
            "class": CLASS_NAMES[cls_id - 1],
            "score": score
        })

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    for r in results:
        x, y, w, h = r["box"]
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(x, y - 5, f"{r['class']} {r['score']:.2f}",
                color='white', fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5))
    ax.set_title(f"R-CNN: {len(results)} detections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    for r in results:
        print(f"  {r['class']:15s} | score={r['score']:.4f} | box={r['box']}")

    return results



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Khởi tạo và sanity check ──────────────────────────────────
    model = RCNN(num_classes=NUM_CLASSES)

    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Sanity check — Input: {dummy.shape} → Output: {out.shape}")
    # Kỳ vọng: (4, 7)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    trained_model = train_rcnn(model, num_epochs=3, lr=1e-4, max_proposals=100)

    save_path = "/content/drive/My Drive/RCNN/rcnn_garbage.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model đã lưu vào {save_path}")

   
