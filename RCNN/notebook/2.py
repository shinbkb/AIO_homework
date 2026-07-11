import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm


DATA_ROOT   = "/content/data/GARBAGE CLASSIFICATION"
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])



class RoIPooling(nn.Module):
    def __init__(self, output_size=7):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois, img_size):
        img_H, img_W = img_size
        _, C, Hf, Wf = feature_map.shape

        # Tỉ lệ scale: ảnh gốc → feature map
        scale_h = Hf / img_H
        scale_w = Wf / img_W

        pooled_list = []
        for roi in rois:
            x1, y1, x2, y2 = roi

            # Chiếu tọa độ pixel → tọa độ trên feature map
            fx1 = max(0,      min(int(x1 * scale_w), Wf - 1))
            fy1 = max(0,      min(int(y1 * scale_h), Hf - 1))
            fx2 = max(fx1+1,  min(int(x2 * scale_w), Wf))
            fy2 = max(fy1+1,  min(int(y2 * scale_h), Hf))

            # Cắt feature map tại vùng RoI
            roi_feat = feature_map[:, :, fy1:fy2, fx1:fx2]  # (1, C, rh, rw)

            # Adaptive Max Pool → output_size × output_size
            pooled = F.adaptive_max_pool2d(roi_feat, self.output_size)  # (1, C, 7, 7)
            pooled_list.append(pooled.squeeze(0))                        # (C, 7, 7)

        return torch.stack(pooled_list, dim=0)  # (N, C, 7, 7)



# FAST RCNN MODEL
class FastRCNN(nn.Module):
    def __init__(self, num_classes, roi_size=7):
        super(FastRCNN, self).__init__()
        self.num_classes = num_classes
        self.roi_size    = roi_size

        # Backbone: ResNet18 đến layer4 (bỏ avgpool + fc)
        # Output: (1, 512, H/32, W/32)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # RoI Pooling
        self.roi_pool = RoIPooling(output_size=roi_size)

        # Sau RoI Pooling flatten: 512 × 7 × 7 = 25088
        flatten_dim = 512 * roi_size * roi_size

        # Shared FC layers
        self.shared_fc = nn.Sequential(
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # cls_head: phân loại (có background)
        self.cls_head = nn.Linear(4096, num_classes + 1)

        # bbox_head: hồi quy offset (dx, dy, dw, dh) cho mỗi class
        self.bbox_head = nn.Linear(4096, 4 * (num_classes + 1))

    def extract_features(self, img_tensor):
        with torch.no_grad():
            return self.backbone(img_tensor)   # (1, 512, Hf, Wf)

    def forward(self, feature_map, rois, img_size):
        pooled = self.roi_pool(feature_map, rois, img_size)   # (N, 512, 7, 7)
        pooled = pooled.view(pooled.size(0), -1)              # (N, 25088)
        shared = self.shared_fc(pooled)                       # (N, 4096)
        cls_scores  = self.cls_head(shared)                   # (N, C+1)
        bbox_deltas = self.bbox_head(shared)                  # (N, 4*(C+1))
        return cls_scores, bbox_deltas


def encode_bbox(proposals, gt_boxes):
    
    Px = (proposals[:, 0] + proposals[:, 2]) / 2
    Py = (proposals[:, 1] + proposals[:, 3]) / 2
    Pw = proposals[:, 2] - proposals[:, 0] + 1e-6
    Ph = proposals[:, 3] - proposals[:, 1] + 1e-6

    Gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    Gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    Gw = gt_boxes[:, 2] - gt_boxes[:, 0] + 1e-6
    Gh = gt_boxes[:, 3] - gt_boxes[:, 1] + 1e-6

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = torch.log(Gw / Pw)
    th = torch.log(Gh / Ph)

    return torch.stack([tx, ty, tw, th], dim=1)  # (N, 4)


def decode_bbox(proposals, bbox_deltas, cls_ids):
    Px = (proposals[:, 0] + proposals[:, 2]) / 2
    Py = (proposals[:, 1] + proposals[:, 3]) / 2
    Pw = proposals[:, 2] - proposals[:, 0]
    Ph = proposals[:, 3] - proposals[:, 1]

    decoded = []
    for i in range(len(proposals)):
        offset = cls_ids[i].item() * 4
        tx = bbox_deltas[i, offset + 0]
        ty = bbox_deltas[i, offset + 1]
        tw = bbox_deltas[i, offset + 2]
        th = bbox_deltas[i, offset + 3]

        Gx = tx * Pw[i] + Px[i]
        Gy = ty * Ph[i] + Py[i]
        Gw = torch.exp(tw) * Pw[i]
        Gh = torch.exp(th) * Ph[i]

        x1 = Gx - Gw / 2
        y1 = Gy - Gh / 2
        x2 = Gx + Gw / 2
        y2 = Gy + Gh / 2
        decoded.append([x1.item(), y1.item(), x2.item(), y2.item()])

    return torch.tensor(decoded)


def get_proposals(img_np, max_proposals=100):
    _, regions = selectivesearch.selective_search(img_np, scale=500, min_size=20)
    rois, seen = [], set()
    for r in regions:
        x, y, w, h = r['rect']
        if w < 20 or h < 20 or (x,y,w,h) in seen:
            continue
        seen.add((x, y, w, h))
        rois.append([x, y, x+w, y+h])
        if len(rois) >= max_proposals:
            break
    return rois


def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = max(0,     int((cx - w/2) * img_w))
    y1 = max(0,     int((cy - h/2) * img_h))
    x2 = min(img_w, int((cx + w/2) * img_w))
    y2 = min(img_h, int((cy + h/2) * img_h))
    return x1, y1, x2, y2


def train_fast_rcnn(model, num_epochs=3, lr=1e-4, max_proposals=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cls_crit  = nn.CrossEntropyLoss()
    bbox_crit = nn.SmoothL1Loss()   # Smooth L1 — đúng theo paper Fast RCNN

    img_paths = (sorted(glob.glob(os.path.join(DATA_ROOT, "train/images/*.jpg")))
               + sorted(glob.glob(os.path.join(DATA_ROOT, "train/images/*.jpeg"))))
    label_dir = os.path.join(DATA_ROOT, "train/labels")
    print(f"Số ảnh train: {len(img_paths)}")

    model.train()
    for epoch in range(num_epochs):
        total_cls, total_bbox = 0.0, 0.0
        total_correct, total_samples = 0, 0

        pbar = tqdm(img_paths, desc=f"Epoch {epoch+1}/{num_epochs}", unit="img")

        for img_path in pbar:
            img_pil = Image.open(img_path).convert("RGB")
            img_w, img_h = img_pil.size

            # Load GT boxes từ YOLO label
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, stem + ".txt")
            gt_boxes, gt_labels = [], []
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0]) + 1
                        x1, y1, x2, y2 = yolo_to_xyxy(*map(float, parts[1:5]), img_w, img_h)
                        if x2 > x1 and y2 > y1:
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_labels.append(cls_id)

            if not gt_boxes:
                continue

            # Selective Search → RoIs
            rois = get_proposals(np.array(img_pil), max_proposals)
            if not rois:
                continue

            # IoU matching → gán nhãn cho từng RoI
            roi_labels, roi_gt_boxes = [], []
            for roi in rois:
                best_iou, best_label, best_gt = 0, 0, roi
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    iou = compute_iou(roi, gt_box)
                    if iou > best_iou:
                        best_iou, best_label, best_gt = iou, gt_label, gt_box
                if best_iou >= 0.5:
                    roi_labels.append(best_label)
                    roi_gt_boxes.append(best_gt)
                elif best_iou <= 0.3:
                    roi_labels.append(0)
                    roi_gt_boxes.append(roi)
                # bỏ qua vùng mơ hồ

            if not roi_labels:
                continue

            # Forward — extract feature map 1 lần duy nhất cho cả ảnh
            img_tensor  = TRANSFORM(img_pil).unsqueeze(0).to(device)
            img_size    = (img_pil.height, img_pil.width)
            feature_map = model.extract_features(img_tensor)

            valid_rois = rois[:len(roi_labels)]
            cls_scores, bbox_deltas = model(feature_map, valid_rois, img_size)

            labels_t   = torch.tensor(roi_labels).to(device)
            gt_boxes_t = torch.tensor(roi_gt_boxes, dtype=torch.float32)
            rois_t     = torch.tensor(valid_rois,   dtype=torch.float32)

            # Classification Loss
            cls_loss = cls_crit(cls_scores, labels_t)

            # Bounding Box Regression Loss (chỉ tính trên positive)
            pos_mask  = labels_t > 0
            bbox_loss = torch.tensor(0.0, device=device)
            if pos_mask.sum() > 0:
                pos_deltas = bbox_deltas[pos_mask]
                pos_labels = labels_t[pos_mask]
                pos_gt     = gt_boxes_t[pos_mask].to(device)
                pos_rois   = rois_t[pos_mask].to(device)

                targets  = encode_bbox(pos_rois, pos_gt)             # (Npos, 4)
                batch_idx = torch.arange(pos_labels.size(0))
                start     = pos_labels * 4
                pred_deltas = torch.stack([
                    pos_deltas[batch_idx, start + k] for k in range(4)
                ], dim=1)                                             # (Npos, 4)

                bbox_loss = bbox_crit(pred_deltas, targets)

            # Multi-task Loss = cls_loss + bbox_loss
            loss = cls_loss + bbox_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = cls_scores.argmax(dim=1)
            total_correct += (preds == labels_t).sum().item()
            total_samples += len(labels_t)
            total_cls     += cls_loss.item()
            total_bbox    += bbox_loss.item()

            pbar.set_postfix({
                "cls" : f"{total_cls  / max(total_samples,1):.4f}",
                "bbox": f"{total_bbox / max(total_samples,1):.4f}",
                "acc" : f"{total_correct / max(total_samples,1):.3f}"
            })

        print(f"✅ Epoch {epoch+1}/{num_epochs} done\n")

    return model


def fast_rcnn_inference(model, image_path, device, threshold=0.5, max_proposals=200):
    model.eval()
    model.to(device)

    img_pil  = Image.open(image_path).convert("RGB")
    img_np   = np.array(img_pil)
    rois     = get_proposals(img_np, max_proposals)

    if not rois:
        print("Không tìm thấy proposals!")
        return []

    img_tensor  = TRANSFORM(img_pil).unsqueeze(0).to(device)
    img_size    = (img_pil.height, img_pil.width)

    with torch.no_grad():
        feature_map          = model.extract_features(img_tensor)
        cls_scores, bbox_deltas = model(feature_map, rois, img_size)

    probs              = torch.softmax(cls_scores, dim=1)
    scores, pred_cls   = probs.max(dim=1)
    decoded_boxes      = decode_bbox(
        torch.tensor(rois, dtype=torch.float32), bbox_deltas.cpu(), pred_cls.cpu()
    )

    results = []
    for i in range(len(rois)):
        cls_id = pred_cls[i].item()
        score  = scores[i].item()
        if cls_id == 0 or score < threshold:
            continue
        x1, y1, x2, y2 = decoded_boxes[i].tolist()
        results.append({
            "box"  : (int(x1), int(y1), int(x2-x1), int(y2-y1)),
            "class": CLASS_NAMES[cls_id - 1],
            "score": score
        })

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_np)
    for r in results:
        x, y, w, h = r["box"]
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none'))
        ax.text(x, y - 5, f"{r['class']} {r['score']:.2f}",
                color='white', fontsize=8,
                bbox=dict(facecolor='green', alpha=0.5))
    ax.set_title(f"Fast R-CNN: {len(results)} detections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    for r in results:
        print(f"  {r['class']:15s} | score={r['score']:.4f} | box={r['box']}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FastRCNN(num_classes=NUM_CLASSES)

    # Sanity check
    dummy_img  = torch.randn(1, 3, 600, 800).to(device)
    dummy_rois = [[50, 30, 200, 180], [300, 100, 500, 400], [10, 400, 150, 580]]
    img_size   = (600, 800)

    model.to(device)
    feat_map            = model.extract_features(dummy_img)
    cls_out, bbox_out   = model(feat_map, dummy_rois, img_size)
    print(f"Feature map : {feat_map.shape}")        # (1, 512, 19, 25)
    print(f"cls_scores  : {cls_out.shape}")         # (3, 7)
    print(f"bbox_deltas : {bbox_out.shape}")         # (3, 28)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    trained_model = train_fast_rcnn(model, num_epochs=3, lr=1e-4, max_proposals=100)

    save_path = "/content/drive/My Drive/RCNN/fast_rcnn.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model đã lưu vào {save_path}")
