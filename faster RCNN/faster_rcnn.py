import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/content/data/GARBAGE CLASSIFICATION"
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class RoIPooling(nn.Module):
    def __init__(self, output_size=7):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois, img_size):
        img_H, img_W = img_size
        _, C, Hf, Wf = feature_map.shape

        scale_h = Hf / img_H
        scale_w = Wf / img_W

        pooled_list = []
        for roi in rois:
            x1, y1, x2, y2 = roi
            fx1 = max(0, min(int(x1 * scale_w), Wf - 1))
            fy1 = max(0, min(int(y1 * scale_h), Hf - 1))
            fx2 = max(fx1 + 1, min(int(x2 * scale_w), Wf))
            fy2 = max(fy1 + 1, min(int(y2 * scale_h), Hf))

            roi_feat = feature_map[:, :, fy1:fy2, fx1:fx2]
            pooled = F.adaptive_max_pool2d(roi_feat, self.output_size)
            pooled_list.append(pooled.squeeze(0))

        return torch.stack(pooled_list, dim=0)


def compute_iou_batch(boxes1, boxes2):
    """
    Tính chỉ số IoU giữa hai nhóm boxes (vectorized).
    boxes1: tensor kích thước (N, 4)
    boxes2: tensor kích thước (M, 4)
    """
    b1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    b2 = boxes2.unsqueeze(0)  # (1, M, 4)

    x1 = torch.max(b1[:, :, 0], b2[:, :, 0])
    y1 = torch.max(b1[:, :, 1], b2[:, :, 1])
    x2 = torch.min(b1[:, :, 2], b2[:, :, 2])
    y2 = torch.min(b1[:, :, 3], b2[:, :, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (b1[:, :, 2] - b1[:, :, 0]) * (b1[:, :, 3] - b1[:, :, 1])
    area2 = (b2[:, :, 2] - b2[:, :, 0]) * (b2[:, :, 3] - b2[:, :, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def encode_bbox(proposals, gt_boxes):
    """ Mã hóa tọa độ tuyệt đối sang dạng delta (dx, dy, dw, dh) """
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
    return torch.stack([tx, ty, tw, th], dim=1)

def decode_bbox(proposals, bbox_deltas, cls_ids):
    """ Giải mã tọa độ từ delta và proposal về tọa độ tuyệt đối """
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

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """ Chuyển tọa độ YOLO định dạng (center_x, center_y, width, height) sang (x1, y1, x2, y2) """
    x1 = max(0, int((cx - w/2) * img_w))
    y1 = max(0, int((cy - h/2) * img_h))
    x2 = min(img_w, int((cx + w/2) * img_w))
    y2 = min(img_h, int((cy + h/2) * img_h))
    return x1, y1, x2, y2


class RPN(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, num_anchors=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.cls_head = nn.Conv2d(mid_channels, num_anchors * 2, 1)  # Phân loại có vật thể hay không
        self.reg_head = nn.Conv2d(mid_channels, num_anchors * 4, 1)  # Hồi quy offset cho anchor

    def forward(self, x):
        h = F.relu(self.conv(x))
        rpn_cls_scores = self.cls_head(h)
        rpn_bbox_pred = self.reg_head(h)

        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        return rpn_cls_scores, rpn_bbox_pred

def generate_anchors(grid_h, grid_w, img_h, img_w, stride=32):
    """ Tạo anchors với 3 kích thước và 3 tỷ lệ khung hình tại mỗi pixel của feature map """
    scales = [128, 256, 512]
    ratios = [0.5, 1.0, 2.0]
    anchors = []
    for y in range(grid_h):
        cy = y * stride + stride / 2.0
        for x in range(grid_w):
            cx = x * stride + stride / 2.0
            for scale in scales:
                for ratio in ratios:
                    h_a = scale * np.sqrt(ratio)
                    w_a = scale / np.sqrt(ratio)
                    anchors.append([cx - w_a/2.0, cy - h_a/2.0, cx + w_a/2.0, cy + h_a/2.0])
    return torch.tensor(anchors, dtype=torch.float32)

def get_proposals_from_rpn(anchors, rpn_cls_scores, rpn_bbox_pred, img_size, is_training=True):
    """ Sử dụng dự đoán của RPN, giải mã tọa độ và chạy NMS để lấy các proposals tốt nhất """
    img_h, img_w = img_size
    probs = F.softmax(rpn_cls_scores[0], dim=1)
    fg_scores = probs[:, 1]

    Px = (anchors[:, 0] + anchors[:, 2]) / 2
    Py = (anchors[:, 1] + anchors[:, 3]) / 2
    Pw = anchors[:, 2] - anchors[:, 0] + 1e-6
    Ph = anchors[:, 3] - anchors[:, 1] + 1e-6

    tx, ty, tw, th = rpn_bbox_pred[0].unbind(dim=1)
    
    # Giới hạn biên độ dự đoán dw, dh để tránh lỗi tràn số khi tính hàm mũ (torch.exp)
    tw = torch.clamp(tw, min=-10.0, max=10.0)
    th = torch.clamp(th, min=-10.0, max=10.0)

    Gx = tx * Pw + Px
    Gy = ty * Ph + Py
    Gw = torch.exp(tw) * Pw
    Gh = torch.exp(th) * Ph

    proposals = torch.stack([
        torch.clamp(Gx - Gw/2, 0, img_w),
        torch.clamp(Gy - Gh/2, 0, img_h),
        torch.clamp(Gx + Gw/2, 0, img_w),
        torch.clamp(Gy + Gh/2, 0, img_h)
    ], dim=1)

    # Loại bỏ các vùng quá nhỏ (nhỏ hơn 16x16)
    w, h = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
    keep = (w >= 16) & (h >= 16)
    proposals, fg_scores = proposals[keep], fg_scores[keep]

    if len(proposals) == 0:
        return torch.tensor([], dtype=torch.float32, device=anchors.device)

    pre_nms_topN = 12000 if is_training else 6000
    post_nms_topN = 2000 if is_training else 300

    order = torch.argsort(fg_scores, descending=True)[:pre_nms_topN]
    proposals, fg_scores = proposals[order], fg_scores[order]

    keep_indices = nms(proposals, fg_scores, iou_threshold=0.7)[:post_nms_topN]
    return proposals[keep_indices]


def get_rpn_targets(anchors, gt_boxes, img_size):
    """ Gán nhãn cho các anchor dựa trên IoU với Ground Truth để tính loss RPN """
    num_anchors = anchors.size(0)
    labels = torch.empty(num_anchors, dtype=torch.long, device=anchors.device).fill_(-1)

    ious = compute_iou_batch(anchors, gt_boxes)
    max_ious, argmax_ious = ious.max(dim=1)
    gt_max_ious, _ = ious.max(dim=0)

    # Đảm bảo mỗi GT box có ít nhất 1 anchor tương ứng
    for i in range(gt_boxes.size(0)):
        labels[ious[:, i] == gt_max_ious[i]] = 1

    labels[max_ious >= 0.7] = 1
    labels[max_ious < 0.3] = 0

    # Loại bỏ anchor có tâm hoặc diện tích vượt ra rìa ảnh
    img_h, img_w = img_size
    inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= img_w) & (anchors[:, 3] <= img_h)
    labels[~inside] = -1

    # Cân bằng tỷ lệ positive và negative (tối đa 128 mỗi nhóm)
    pos_idx = torch.where(labels == 1)[0]
    if len(pos_idx) > 128:
        labels[pos_idx[torch.randperm(len(pos_idx))[128:]]] = -1

    neg_idx = torch.where(labels == 0)[0]
    num_neg = 256 - (labels == 1).sum().item()
    if len(neg_idx) > num_neg:
        labels[neg_idx[torch.randperm(len(neg_idx))[num_neg:]]] = -1

    pos_anchor_idx = torch.where(labels == 1)[0]
    bbox_targets = encode_bbox(anchors[pos_anchor_idx], gt_boxes[argmax_ious[pos_anchor_idx]])
    return labels, bbox_targets, pos_anchor_idx

def sample_proposals(proposals, gt_boxes, gt_labels, num_samples=128, pos_ratio=0.25):
    """ Gom proposals thu được từ RPN kết hợp với GT boxes, gán nhãn và lọc cân bằng mẫu """
    ious = compute_iou_batch(proposals, gt_boxes)
    max_ious, argmax_ious = ious.max(dim=1)

    labels = torch.zeros(proposals.size(0), dtype=torch.long, device=proposals.device)
    matched_gt_boxes = proposals.clone()

    pos_mask = max_ious >= 0.5
    labels[pos_mask] = gt_labels[argmax_ious[pos_mask]]
    matched_gt_boxes[pos_mask] = gt_boxes[argmax_ious[pos_mask]]

    bg_mask = (max_ious < 0.5) & (max_ious >= 0.1)
    labels[bg_mask] = 0

    pos_idx = torch.where(labels > 0)[0]
    num_pos = int(num_samples * pos_ratio)
    if len(pos_idx) > num_pos:
        pos_idx = pos_idx[torch.randperm(len(pos_idx))[:num_pos]]

    neg_idx = torch.where(labels == 0)[0]
    num_neg = num_samples - len(pos_idx)
    if len(neg_idx) > num_neg:
        neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]

    keep = torch.cat([pos_idx, neg_idx])
    return proposals[keep], labels[keep], matched_gt_boxes[keep]


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, roi_size=7):
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.roi_size = roi_size

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.rpn = RPN(in_channels=512, mid_channels=512, num_anchors=9)
        self.roi_pool = RoIPooling(output_size=roi_size)

        flatten_dim = 512 * roi_size * roi_size
        self.shared_fc = nn.Sequential(
            nn.Linear(flatten_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.cls_head = nn.Linear(4096, num_classes + 1)
        self.bbox_head = nn.Linear(4096, 4 * (num_classes + 1))

    def extract_features(self, img_tensor):
        return self.backbone(img_tensor)

    def forward(self, feature_map, rois, img_size):
        pooled = self.roi_pool(feature_map, rois, img_size)
        pooled = pooled.view(pooled.size(0), -1)
        shared = self.shared_fc(pooled)
        cls_scores = self.cls_head(shared)
        bbox_deltas = self.bbox_head(shared)
        return cls_scores, bbox_deltas


def train_faster_rcnn(model, num_epochs=3, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Huấn luyện trên thiết bị: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    img_paths = (sorted(glob.glob(os.path.join(DATA_ROOT, "train/images/*.jpg")))
                 + sorted(glob.glob(os.path.join(DATA_ROOT, "train/images/*.jpeg"))))
    label_dir = os.path.join(DATA_ROOT, "train/labels")
    print(f"Tổng số ảnh huấn luyện: {len(img_paths)}")

    model.train()
    for epoch in range(num_epochs):
        total_rpn_cls, total_rpn_box = 0.0, 0.0
        total_det_cls, total_det_box = 0.0, 0.0
        pbar = tqdm(img_paths, desc=f"Epoch {epoch+1}/{num_epochs}", unit="img")

        for img_path in pbar:
            img_pil = Image.open(img_path).convert("RGB")
            img_w, img_h = img_pil.size

            # Đọc nhãn GT YOLO
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
            if not gt_boxes: continue

            # Chuẩn bị Tensor dữ liệu đưa lên GPU
            img_tensor = TRANSFORM(img_pil).unsqueeze(0).to(device)
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            gt_labels_t = torch.tensor(gt_labels, dtype=torch.long, device=device)

            # 1. Trích xuất đặc trưng Backbone & Dự báo RPN
            feature_map = model.extract_features(img_tensor)
            rpn_cls_scores, rpn_bbox_pred = model.rpn(feature_map)

            # 2. Tạo Anchors và xác định nhãn tiêu chuẩn cho RPN
            Hf, Wf = feature_map.shape[2], feature_map.shape[3]
            anchors = generate_anchors(Hf, Wf, img_h, img_w).to(device)
            rpn_labels, rpn_bbox_targets, pos_anchor_idx = get_rpn_targets(anchors, gt_boxes_t, (img_h, img_w))

            # Tính toán loss RPN
            loss_rpn_cls = F.cross_entropy(rpn_cls_scores[0], rpn_labels, ignore_index=-1)
            loss_rpn_box = torch.tensor(0.0, device=device)
            if len(pos_anchor_idx) > 0:
                loss_rpn_box = F.smooth_l1_loss(rpn_bbox_pred[0, pos_anchor_idx], rpn_bbox_targets)

            # 3. Tạo Proposals & Chọn lọc mẫu cho bộ phân loại (Detector)
            proposals = get_proposals_from_rpn(anchors, rpn_cls_scores, rpn_bbox_pred, (img_h, img_w), is_training=True)
            if len(proposals) == 0: continue
            sampled_props, sampled_labels, sampled_gt_boxes = sample_proposals(proposals, gt_boxes_t, gt_labels_t)

            # 4. Dự báo của Detector và tính Loss
            cls_scores, bbox_deltas = model(feature_map, sampled_props.tolist(), (img_h, img_w))

            loss_det_cls = F.cross_entropy(cls_scores, sampled_labels)
            loss_det_box = torch.tensor(0.0, device=device)
            pos_mask = sampled_labels > 0
            if pos_mask.sum() > 0:
                pos_deltas = bbox_deltas[pos_mask]
                pos_lbls = sampled_labels[pos_mask]
                targets = encode_bbox(sampled_props[pos_mask], sampled_gt_boxes[pos_mask])

                batch_idx = torch.arange(pos_lbls.size(0))
                start = pos_lbls * 4
                pred_deltas = torch.stack([pos_deltas[batch_idx, start + k] for k in range(4)], dim=1)
                loss_det_box = F.smooth_l1_loss(pred_deltas, targets)

            # Tổng hợp đa mục tiêu loss và tối ưu hóa
            loss = loss_rpn_cls + loss_rpn_box + loss_det_cls + loss_det_box
            optimizer.zero_grad()
            loss.backward()
            
            # Cắt ngắn gradient để chống hiện tượng bùng nổ (Gradient Explosion) gây mất ổn định RPN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            # Lưu lại thông tin loss
            total_rpn_cls += loss_rpn_cls.item()
            total_rpn_box += loss_rpn_box.item()
            total_det_cls += loss_det_cls.item()
            total_det_box += loss_det_box.item()

            pbar.set_postfix({
                "rpn_c": f"{loss_rpn_cls.item():.3f}",
                "rpn_b": f"{loss_rpn_box.item():.3f}",
                "det_c": f"{loss_det_cls.item():.3f}",
                "det_b": f"{loss_det_box.item():.3f}"
            })

        print(f"✅ Epoch {epoch+1} done! rpn_cls={total_rpn_cls/len(img_paths):.4f}, det_cls={total_det_cls/len(img_paths):.4f}\n")
    return model


def draw_boxes(ax, boxes, labels, scores=None, color='red'):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label_text = CLASS_NAMES[labels[i] - 1]
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
        ax.text(x1, y1 - 10, label_text, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0))

def evaluate_and_visualize(model, test_img_paths, device, score_thresh=0.7, iou_thresh=0.5):
    model.eval()
    with torch.no_grad():
        for img_path in test_img_paths:
            img_pil = Image.open(img_path).convert("RGB")
            img_w, img_h = img_pil.size

            # Load Ground Truth
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(DATA_ROOT, "test/labels", stem + ".txt")
            gt_boxes, gt_labels = [], []
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cls_id = int(parts[0]) + 1
                        x1, y1, x2, y2 = yolo_to_xyxy(*map(float, parts[1:5]), img_w, img_h)
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(cls_id)
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32, device=device)
            gt_labels_t = torch.tensor(gt_labels, dtype=torch.long, device=device)

            # Khởi tạo ảnh
            img_tensor = TRANSFORM(img_pil).unsqueeze(0).to(device)

            feature_map = model.extract_features(img_tensor)
            rpn_cls_scores, rpn_bbox_pred = model.rpn(feature_map)

            Hf, Wf = feature_map.shape[2], feature_map.shape[3]
            anchors = generate_anchors(Hf, Wf, img_h, img_w).to(device)
            proposals = get_proposals_from_rpn(anchors, rpn_cls_scores, rpn_bbox_pred, (img_h, img_w), is_training=False)

            if len(proposals) == 0:
                print(f"Không tìm thấy proposals cho ảnh: {img_path}")
                continue

            cls_scores, bbox_deltas = model(feature_map, proposals.tolist(), (img_h, img_w))
            cls_probs = F.softmax(cls_scores, dim=1)

            max_probs, pred_labels = torch.max(cls_probs[:, 1:], dim=1)
            pred_labels = pred_labels + 1

            # Lọc theo ngưỡng tự tin
            keep_conf = max_probs >= score_thresh
            proposals_filtered = proposals[keep_conf]
            pred_labels_filtered = pred_labels[keep_conf]
            max_probs_filtered = max_probs[keep_conf]
            bbox_deltas_filtered = bbox_deltas[keep_conf]

            if len(proposals_filtered) == 0:
                print(f"Không có dự đoán nào đạt ngưỡng tự tin cho ảnh: {img_path}")
                continue

            final_boxes_decoded = decode_bbox(proposals_filtered, bbox_deltas_filtered, pred_labels_filtered)
            nms_indices = nms(final_boxes_decoded, max_probs_filtered, iou_threshold=iou_thresh)

            final_boxes = final_boxes_decoded[nms_indices]
            final_labels = pred_labels_filtered[nms_indices]
            final_scores = max_probs_filtered[nms_indices]

            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_pil)
            ax.set_title(f"Dự đoán cho ảnh: {os.path.basename(img_path)}")
            ax.axis('off')

            if len(gt_boxes_t) > 0:
                draw_boxes(ax, gt_boxes_t, gt_labels_t, color='green')

            if len(final_boxes) > 0:
                draw_boxes(ax, final_boxes, final_labels, final_scores, color='red')

            plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FasterRCNN(num_classes=NUM_CLASSES)

    # Khởi chạy train
    trained_model = train_faster_rcnn(model, num_epochs=3, lr=1e-4)

    # Tạo thư mục và lưu trọng số
    save_path = "/content/drive/My Drive/RCNN/faster_rcnn.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f"Lưu model thành công tại: {save_path}")

    # Đánh giá trực quan hóa
    test_img_dir = os.path.join(DATA_ROOT, "test/images")
    test_img_paths = (sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
                      + sorted(glob.glob(os.path.join(test_img_dir, "*.jpeg"))))
    
    print("\nBắt đầu đánh giá model trên 5 ảnh kiểm thử đầu tiên...")
    evaluate_and_visualize(trained_model, test_img_paths[:5], device, score_thresh=0.05)
    print("Đánh giá hoàn tất!")
