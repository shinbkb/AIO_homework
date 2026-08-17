import os
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.ops import roi_align, nms, box_iou

# ======================================================================
# 2. Config
# ======================================================================
class Config:
    """
    Hyperparameters and directory configurations for Faster R-CNN training and evaluation.
    """
    # Dataset Directories
    dataset_dir = "D:/WorkSpace/Learn/AI/dutai_ex/dataset/GARBAGE CLASSIFICATION"
    train_images = os.path.join(dataset_dir, "train/images")
    train_labels = os.path.join(dataset_dir, "train/labels")
    val_images = os.path.join(dataset_dir, "valid/images")
    val_labels = os.path.join(dataset_dir, "valid/labels")
    test_images = os.path.join(dataset_dir, "test/images")
    test_labels = os.path.join(dataset_dir, "test/labels")
    
    # Image Size
    image_size = (416, 416)
    
    # Class Definitions
    # 6 garbage classes (mapped to labels 1-6; label 0 is reserved for background)
    class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    num_classes = len(class_names)
    
    # Backbone Config
    backbone_out_channels = 1024  # ResNet50 Layer3 out channels
    
    # Anchor Generator Config
    anchor_scales = [64, 128, 256]
    anchor_ratios = [0.5, 1.0, 2.0]
    
    # RPN Hyperparameters
    rpn_pre_nms_top_n_train = 12000
    rpn_post_nms_top_n_train = 2000
    rpn_pre_nms_top_n_test = 6000
    rpn_post_nms_top_n_test = 300
    rpn_nms_thresh = 0.7
    rpn_batch_size = 256
    rpn_fg_fraction = 0.5
    
    # RoI Head / Fast R-CNN Hyperparameters
    roi_output_size = (7, 7)
    roi_spatial_scale = 1.0 / 16.0  # Stride 16 of Layer3 features
    fast_rcnn_batch_size = 128
    fast_rcnn_fg_fraction = 0.25
    fast_rcnn_fg_iou_thresh = 0.5
    fast_rcnn_bg_iou_thresh_hi = 0.5
    fast_rcnn_bg_iou_thresh_lo = 0.1
    
    # Post-processing / Inference Hyperparameters
    final_score_thresh = 0.05             # Threshold for mAP evaluation
    final_inference_score_thresh = 0.5    # Threshold for demo visualization
    final_nms_thresh = 0.3                # NMS threshold for final detections
    
    # Training Parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    epochs = 10
    batch_size = 4
    num_workers = 0  # set to 0 for Windows compatibility
    save_path = "checkpoints/faster_rcnn_garbage.pth"


# ======================================================================
# 3. Dataset
# ======================================================================
class GarbageDataset(Dataset):
    """
    Dataset class for Garbage Classification in YOLO format.
    Parses bounding boxes, converts relative coords to absolute pixel coordinates,
    and shifts class labels by +1 to reserve index 0 for the Background.
    """
    def __init__(self, images_dir: str, labels_dir: str, image_size: Tuple[int, int], transforms: Optional[Any] = None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.transforms = transforms
        
        # Get all valid image names
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image in RGB
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Parse corresponding labels if present
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.labels_dir, base_name + ".txt")
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Shift label index (+1) because 0 represents Background
                        class_label = class_id + 1
                        
                        # YOLO relative coords to absolute pixels
                        x_center = float(parts[1]) * orig_w
                        y_center = float(parts[2]) * orig_h
                        w = float(parts[3]) * orig_w
                        h = float(parts[4]) * orig_h
                        
                        xmin = x_center - w / 2.0
                        ymin = y_center - h / 2.0
                        xmax = x_center + w / 2.0
                        ymax = y_center + h / 2.0
                        
                        # Rescale coordinates to target image size
                        target_w, target_h = self.image_size
                        scale_x = target_w / orig_w
                        scale_y = target_h / orig_h
                        
                        xmin = xmin * scale_x
                        ymin = ymin * scale_y
                        xmax = xmax * scale_x
                        ymax = ymax * scale_y
                        
                        # Clip to boundaries
                        xmin = max(0.0, min(xmin, float(target_w)))
                        ymin = max(0.0, min(ymin, float(target_h)))
                        xmax = max(0.0, min(xmax, float(target_w)))
                        ymax = max(0.0, min(ymax, float(target_h)))
                        
                        # Keep only valid boxes
                        if xmax > xmin + 1.0 and ymax > ymin + 1.0:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(class_label)
                            
        # Wrap outputs in PyTorch tensors
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor
        }
        
        # Apply standard transforms
        if self.transforms:
            img = self.transforms(img)
        else:
            default_transforms = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img = default_transforms(img)
            
        return img, target

def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Collate function to bundle images and variable-length labels in a batch.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


# ======================================================================
# 4. Backbone
# ======================================================================
class ResNetBackbone(nn.Module):
    """
    ResNet50 Backbone feature extractor. Truncated at Layer 3 (Conv4_x) to 
    output 1024 channels with stride 16. Freezes early layers to improve convergence.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load backbone model
        if hasattr(torchvision.models, 'ResNet50_Weights'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
        else:
            resnet = torchvision.models.resnet50(pretrained=pretrained)
            
        # Extract layers up to layer3 (conv4_x stage)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        # Freeze parameters of early stages
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
            
        # Put frozen modules explicitly in eval mode
        self.conv1.eval()
        self.bn1.eval()
        self.layer1.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Ensure frozen layers stay in eval mode during training
        self.conv1.eval()
        self.bn1.eval()
        self.layer1.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ======================================================================
# 5. Region Proposal Network (RPN)
# ======================================================================
def generate_anchors(feature_h: int, feature_w: int, stride: int, 
                     scales: List[float], ratios: List[float], device: torch.device) -> torch.Tensor:
    """
    Generates reference anchors for all spatial locations on the feature map.
    """
    num_anchors = len(scales) * len(ratios)
    
    # 1. Base anchors centered around (0, 0)
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            h = scale / math.sqrt(ratio)
            w = scale * math.sqrt(ratio)
            base_anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
            
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device) # [num_anchors, 4]
    
    # 2. Compute shifts across the grid
    shift_x = torch.arange(0, feature_w, dtype=torch.float32, device=device) * stride + stride / 2.0
    shift_y = torch.arange(0, feature_h, dtype=torch.float32, device=device) * stride + stride / 2.0
    
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) # [H_feat * W_feat, 4]
    
    # 3. Add base anchors and shifts
    all_anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0) # [H*W, num_anchors, 4]
    return all_anchors.reshape(-1, 4) # [H*W*num_anchors, 4]

def box_encode(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Encodes coordinates to bounding box delta targets (tx, ty, tw, th).
    """
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + 0.5 * w_a
    y_a = anchors[:, 1] + 0.5 * h_a
    
    w_g = boxes[:, 2] - boxes[:, 0]
    h_g = boxes[:, 3] - boxes[:, 1]
    x_g = boxes[:, 0] + 0.5 * w_g
    y_g = boxes[:, 1] + 0.5 * h_g
    
    w_g = torch.clamp(w_g, min=1.0)
    h_g = torch.clamp(h_g, min=1.0)
    
    tx = (x_g - x_a) / w_a
    ty = (y_g - y_a) / h_a
    tw = torch.log(w_g / w_a)
    th = torch.log(h_g / h_a)
    
    return torch.stack([tx, ty, tw, th], dim=1)

def box_decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Decodes bounding box delta targets back to coordinates (xmin, ymin, xmax, ymax).
    """
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + 0.5 * w_a
    y_a = anchors[:, 1] + 0.5 * h_a
    
    tx = deltas[:, 0]
    ty = deltas[:, 1]
    tw = deltas[:, 2]
    th = deltas[:, 3]
    
    # Restrict exponentiation to prevent overflow
    tw = torch.clamp(tw, max=10.0)
    th = torch.clamp(th, max=10.0)
    
    w_pred = torch.exp(tw) * w_a
    h_pred = torch.exp(th) * h_a
    x_pred = tx * w_a + x_a
    y_pred = ty * h_a + y_a
    
    xmin = x_pred - 0.5 * w_pred
    ymin = y_pred - 0.5 * h_pred
    xmax = x_pred + 0.5 * w_pred
    ymax = y_pred + 0.5 * h_pred
    
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)

class RegionProposalNetwork(nn.Module):
    """
    Region Proposal Network (RPN) outputting classification logits (objectness)
    and spatial deltas for generated anchors.
    """
    def __init__(self, in_channels: int, num_anchors: int, config: Config):
        super().__init__()
        self.config = config
        self.num_anchors = num_anchors
        
        # sliding window features
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Cls and Reg branches
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
        
        # Init weights
        for layer in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, feat_h, feat_w = features.shape
        device = features.device
        
        t = self.relu(self.conv(features))
        
        # [B, num_anchors*2, H_feat, W_feat] -> [B, H_feat*W_feat*num_anchors, 2]
        cls_logits = self.cls_layer(t)
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        
        # [B, num_anchors*4, H_feat, W_feat] -> [B, H_feat*W_feat*num_anchors, 4]
        bbox_deltas = self.reg_layer(t)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Generate anchors
        anchors = generate_anchors(
            feature_h=feat_h,
            feature_w=feat_w,
            stride=16,
            scales=self.config.anchor_scales,
            ratios=self.config.anchor_ratios,
            device=device
        )
        
        losses = {}
        if self.training and targets is not None:
            losses = self.compute_loss(cls_logits, bbox_deltas, anchors, targets)
            
        return cls_logits, bbox_deltas, anchors, losses

    def compute_loss(self, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, 
                     anchors: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        device = cls_logits.device
        batch_size = cls_logits.shape[0]
        
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        
        for i in range(batch_size):
            gt_boxes = targets[i]["boxes"]
            img_cls_logits = cls_logits[i]
            img_bbox_deltas = bbox_deltas[i]
            
            if len(gt_boxes) == 0:
                # If no targets exist, all anchors are background (class 0)
                labels = torch.zeros(len(anchors), dtype=torch.int64, device=device)
                sampled_idx = torch.randperm(len(anchors), device=device)[:self.config.rpn_batch_size]
                total_cls_loss += F.cross_entropy(img_cls_logits[sampled_idx], labels[sampled_idx])
                total_reg_loss += torch.tensor(0.0, device=device)
                continue
                
            # 1. IoU Matrix [num_anchors, num_gts]
            ious = box_iou(anchors, gt_boxes)
            
            max_ious_per_anchor, matched_gt_idx = ious.max(dim=1)
            max_ious_per_gt, matched_anchor_idx = ious.max(dim=0)
            
            # Init label state (-1 = ignore)
            labels = torch.full((len(anchors),), -1, dtype=torch.int64, device=device)
            
            # Negatives: IoU < 0.3
            labels[max_ious_per_anchor < 0.3] = 0
            
            # Positives: IoU >= 0.7
            labels[max_ious_per_anchor >= 0.7] = 1
            
            # Positives: highest IoU anchor per GT box
            for gt_idx, anchor_idx in enumerate(matched_anchor_idx):
                labels[anchor_idx] = 1
                
            # 2. Sample anchor sets (up to 128 positive, rest negative)
            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where(labels == 0)[0]
            
            num_pos = min(len(pos_idx), int(self.config.rpn_batch_size * self.config.rpn_fg_fraction))
            num_neg = self.config.rpn_batch_size - num_pos
            num_neg = min(len(neg_idx), num_neg)
            
            sampled_pos = pos_idx[torch.randperm(len(pos_idx), device=device)[:num_pos]] if len(pos_idx) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
            sampled_neg = neg_idx[torch.randperm(len(neg_idx), device=device)[:num_neg]] if len(neg_idx) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
            sampled_idx = torch.cat([sampled_pos, sampled_neg])
            
            # 3. Classification Loss
            loss_cls = F.cross_entropy(img_cls_logits[sampled_idx], labels[sampled_idx])
            
            # 4. Box Regression Loss
            if len(sampled_pos) > 0:
                matched_gt_boxes = gt_boxes[matched_gt_idx[sampled_pos]]
                targets_reg = box_encode(matched_gt_boxes, anchors[sampled_pos])
                loss_reg = F.smooth_l1_loss(img_bbox_deltas[sampled_pos], targets_reg, reduction='mean')
            else:
                loss_reg = torch.tensor(0.0, device=device)
                
            total_cls_loss += loss_cls
            total_reg_loss += loss_reg
            
        return {
            "loss_rpn_cls": total_cls_loss / batch_size,
            "loss_rpn_reg": total_reg_loss / batch_size
        }


# ======================================================================
# 6. Proposal Generator
# ======================================================================
class ProposalGenerator:
    """
    Decodes predictions, clips to boundaries, runs NMS, and returns RoIs.
    """
    def __init__(self, config: Config):
        self.config = config

    @torch.no_grad()
    def __call__(self, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, 
                 anchors: torch.Tensor, is_training: bool) -> List[torch.Tensor]:
        device = cls_logits.device
        batch_size = cls_logits.shape[0]
        
        pre_nms_top_n = self.config.rpn_pre_nms_top_n_train if is_training else self.config.rpn_pre_nms_top_n_test
        post_nms_top_n = self.config.rpn_post_nms_top_n_train if is_training else self.config.rpn_post_nms_top_n_test
        
        # Softmax to get foreground probability scores
        scores = F.softmax(cls_logits, dim=-1)[:, :, 1]
        
        proposals = []
        for i in range(batch_size):
            img_scores = scores[i]
            img_deltas = bbox_deltas[i]
            
            # Decode RPN predictions
            decoded_boxes = box_decode(img_deltas, anchors)
            
            # Clip boundaries
            w_img, h_img = self.config.image_size
            decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
            decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
            decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
            decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
            
            # Filter boxes too small (width or height < 1)
            ws = decoded_boxes[:, 2] - decoded_boxes[:, 0]
            hs = decoded_boxes[:, 3] - decoded_boxes[:, 1]
            keep = (ws >= 1.0) & (hs >= 1.0)
            
            img_scores = img_scores[keep]
            decoded_boxes = decoded_boxes[keep]
            
            if len(decoded_boxes) == 0:
                # fallback
                proposals.append(anchors[:10].clone())
                continue
                
            # Filter Pre-NMS Top-K
            k = min(len(decoded_boxes), pre_nms_top_n)
            topk_scores, topk_idx = torch.topk(img_scores, k)
            topk_boxes = decoded_boxes[topk_idx]
            
            # Run NMS
            keep_nms = nms(topk_boxes, topk_scores, self.config.rpn_nms_thresh)
            
            # Keep Post-NMS Top-K
            keep_nms = keep_nms[:post_nms_top_n]
            proposals.append(topk_boxes[keep_nms])
            
        return proposals


# ======================================================================
# 7. RoI Head
# ======================================================================
class RoIHead(nn.Module):
    """
    RoI Head handles sampling proposal regions during training 
    and crops region features via RoI Align.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def sample_proposals(self, proposals: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        device = proposals[0].device
        sampled_proposals = []
        sampled_labels = []
        sampled_gt_deltas = []
        
        for i in range(len(proposals)):
            img_proposals = proposals[i]
            gt_boxes = targets[i]["boxes"]
            gt_labels = targets[i]["labels"]
            
            if len(gt_boxes) == 0:
                # All proposals are background
                labels = torch.zeros(len(img_proposals), dtype=torch.int64, device=device)
                perm = torch.randperm(len(img_proposals), device=device)[:self.config.fast_rcnn_batch_size]
                
                sampled_proposals.append(img_proposals[perm])
                sampled_labels.append(labels[perm])
                sampled_gt_deltas.append(torch.zeros((len(perm), 4), dtype=torch.float32, device=device))
                continue
                
            # Concatenate GT boxes to proposal pool to guarantee positive samples
            img_proposals = torch.cat([img_proposals, gt_boxes], dim=0)
            
            # IoU Matrix [num_proposals, num_gts]
            ious = box_iou(img_proposals, gt_boxes)
            max_ious, matched_gt_idx = ious.max(dim=1)
            matched_classes = gt_labels[matched_gt_idx]
            
            labels = torch.full((len(img_proposals),), -1, dtype=torch.int64, device=device)
            
            # Backgrounds: 0.1 <= IoU < 0.5
            bg_mask = (max_ious >= self.config.fast_rcnn_bg_iou_thresh_lo) & (max_ious < self.config.fast_rcnn_bg_iou_thresh_hi)
            labels[bg_mask] = 0
            
            # Foreground: IoU >= 0.5
            fg_mask = max_ious >= self.config.fast_rcnn_fg_iou_thresh
            labels[fg_mask] = matched_classes[fg_mask]
            
            # Sample balanced subsets (e.g. 25% foreground, 75% background)
            fg_idx = torch.where(labels > 0)[0]
            bg_idx = torch.where(labels == 0)[0]
            
            num_fg = min(len(fg_idx), int(self.config.fast_rcnn_batch_size * self.config.fast_rcnn_fg_fraction))
            num_bg = self.config.fast_rcnn_batch_size - num_fg
            num_bg = min(len(bg_idx), num_bg)
            
            sampled_fg = fg_idx[torch.randperm(len(fg_idx), device=device)[:num_fg]] if len(fg_idx) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
            sampled_bg = bg_idx[torch.randperm(len(bg_idx), device=device)[:num_bg]] if len(bg_idx) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
            sampled_idx = torch.cat([sampled_fg, sampled_bg])
            
            img_sampled_proposals = img_proposals[sampled_idx]
            img_sampled_labels = labels[sampled_idx]
            
            # Compute box targets relative to matched GT boxes for positive samples
            img_sampled_deltas = torch.zeros((len(sampled_idx), 4), dtype=torch.float32, device=device)
            if len(sampled_fg) > 0:
                matched_gt_boxes = gt_boxes[matched_gt_idx[sampled_fg]]
                fg_deltas = box_encode(matched_gt_boxes, img_proposals[sampled_fg])
                img_sampled_deltas[:num_fg] = fg_deltas
                
            sampled_proposals.append(img_sampled_proposals)
            sampled_labels.append(img_sampled_labels)
            sampled_gt_deltas.append(img_sampled_deltas)
            
        return sampled_proposals, sampled_labels, sampled_gt_deltas

    def forward(self, features: torch.Tensor, proposals: List[torch.Tensor]) -> torch.Tensor:
        # Crop region features using roi_align
        return roi_align(
            features, 
            proposals, 
            output_size=self.config.roi_output_size, 
            spatial_scale=self.config.roi_spatial_scale
        )


# ======================================================================
# 8. Classification Head
# ======================================================================
class ClassificationHead(nn.Module):
    """
    Standard fully connected classifier head taking cropped features 
    and outputting class logits.
    """
    def __init__(self, in_channels: int, num_classes: int, config: Config):
        super().__init__()
        self.config = config
        
        flat_features = in_channels * config.roi_output_size[0] * config.roi_output_size[1]
        
        self.fc1 = nn.Linear(flat_features, 1024)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        
        self.cls_score = nn.Linear(1024, num_classes + 1)
        
        for layer in [self.fc1, self.fc2, self.cls_score]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(self.relu1(self.fc1(x)))
        shared_features = self.dropout2(self.relu2(self.fc2(x)))
        cls_logits = self.cls_score(shared_features)
        return cls_logits, shared_features


# ======================================================================
# 9. Bounding Box Head
# ======================================================================
class BoundingBoxHead(nn.Module):
    """
    Fast R-CNN Head predicting class-specific spatial deltas for RoIs.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        # Outputs [num_rois, (num_classes + 1) * 4]
        self.bbox_pred = nn.Linear(in_features, (num_classes + 1) * 4)
        
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        return self.bbox_pred(shared_features)


# ======================================================================
# 10. FasterRCNN Wrapper
# ======================================================================
class FasterRCNN(nn.Module):
    """
    Faster R-CNN unified detector orchestrating all modules.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 1. Backbone ResNet50 up to layer3
        self.backbone = ResNetBackbone(pretrained=True)
        
        # 2. Region Proposal Network
        num_anchors = len(config.anchor_scales) * len(config.anchor_ratios)
        self.rpn = RegionProposalNetwork(
            in_channels=config.backbone_out_channels,
            num_anchors=num_anchors,
            config=config
        )
        
        # 3. Proposal Generator
        self.proposal_generator = ProposalGenerator(config=config)
        
        # 4. RoI Head
        self.roi_head = RoIHead(config=config)
        
        # 5. Heads
        self.classification_head = ClassificationHead(
            in_channels=config.backbone_out_channels,
            num_classes=config.num_classes,
            config=config
        )
        self.bbox_head = BoundingBoxHead(
            in_features=1024,
            num_classes=config.num_classes
        )

    def forward(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        device = images.device
        batch_size = images.shape[0]
        
        # Extract features
        features = self.backbone(images)
        
        # RPN output
        rpn_cls_logits, rpn_bbox_deltas, anchors, rpn_losses = self.rpn(features, targets)
        
        # Generate Proposals
        proposals = self.proposal_generator(rpn_cls_logits, rpn_bbox_deltas, anchors, self.training)
        
        if self.training:
            assert targets is not None, "Targets must be provided during training mode"
            
            # Sample Proposals for RoI Heads
            sampled_proposals, sampled_labels, sampled_gt_deltas = self.roi_head.sample_proposals(proposals, targets)
            
            # Crop RoI features
            roi_features = self.roi_head(features, sampled_proposals)
            
            # Predictions
            cls_logits, shared_features = self.classification_head(roi_features)
            bbox_deltas = self.bbox_head(shared_features)
            
            # Flatten target lists across batch
            cat_sampled_labels = torch.cat(sampled_labels, dim=0)
            cat_gt_deltas = torch.cat(sampled_gt_deltas, dim=0)
            
            # Fast R-CNN Classification Loss
            loss_fast_rcnn_cls = F.cross_entropy(cls_logits, cat_sampled_labels)
            
            # Fast R-CNN Box Regression Loss (computed only on positive proposals)
            fg_idx = torch.where(cat_sampled_labels > 0)[0]
            if len(fg_idx) > 0:
                fg_classes = cat_sampled_labels[fg_idx]
                
                # Extract predicted deltas matching target class index
                pred_deltas = bbox_deltas[fg_idx].reshape(-1, self.config.num_classes + 1, 4)
                pred_deltas = pred_deltas[torch.arange(len(fg_idx), device=device), fg_classes]
                
                loss_fast_rcnn_reg = F.smooth_l1_loss(pred_deltas, cat_gt_deltas[fg_idx], reduction='mean')
            else:
                loss_fast_rcnn_reg = torch.tensor(0.0, device=device)
                
            return {
                "loss_rpn_cls": rpn_losses["loss_rpn_cls"],
                "loss_rpn_reg": rpn_losses["loss_rpn_reg"],
                "loss_fast_rcnn_cls": loss_fast_rcnn_cls,
                "loss_fast_rcnn_reg": loss_fast_rcnn_reg
            }
            
        else:
            # Inference mode
            roi_features = self.roi_head(features, proposals)
            cls_logits, shared_features = self.classification_head(roi_features)
            bbox_deltas = self.bbox_head(shared_features)
            
            cls_probs = F.softmax(cls_logits, dim=-1)
            
            # Split features per image
            proposals_count = [len(p) for p in proposals]
            cls_probs_split = torch.split(cls_probs, proposals_count, dim=0)
            bbox_deltas_split = torch.split(bbox_deltas, proposals_count, dim=0)
            
            predictions = []
            w_img, h_img = self.config.image_size
            
            for idx in range(batch_size):
                img_proposals = proposals[idx]
                img_cls_probs = cls_probs_split[idx]
                img_bbox_deltas = bbox_deltas_split[idx]
                
                img_final_boxes = []
                img_final_scores = []
                img_final_labels = []
                
                # Process class-specific detections
                for c in range(1, self.config.num_classes + 1):
                    class_scores = img_cls_probs[:, c]
                    
                    # Filter scores below threshold
                    keep_idx = torch.where(class_scores > self.config.final_score_thresh)[0]
                    if len(keep_idx) == 0:
                        continue
                        
                    keep_scores = class_scores[keep_idx]
                    keep_proposals = img_proposals[keep_idx]
                    keep_deltas = img_bbox_deltas[keep_idx, 4*c : 4*c+4]
                    
                    # Decode coordinates
                    decoded_boxes = box_decode(keep_deltas, keep_proposals)
                    
                    # Clip
                    decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
                    decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
                    decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
                    decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
                    
                    # Run class-wise NMS
                    keep_nms = nms(decoded_boxes, keep_scores, self.config.final_nms_thresh)
                    
                    img_final_boxes.append(decoded_boxes[keep_nms])
                    img_final_scores.append(keep_scores[keep_nms])
                    img_final_labels.append(torch.full((len(keep_nms),), c, dtype=torch.int64, device=device))
                    
                if len(img_final_boxes) > 0:
                    img_pred = {
                        "boxes": torch.cat(img_final_boxes, dim=0),
                        "scores": torch.cat(img_final_scores, dim=0),
                        "labels": torch.cat(img_final_labels, dim=0)
                    }
                else:
                    img_pred = {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device)
                    }
                predictions.append(img_pred)
                
            return predictions


# ======================================================================
# 12. Training Function
# ======================================================================
def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, print_freq: int = 20) -> Dict[str, float]:
    """
    Executes training loop over one epoch.
    """
    model.train()
    running_losses = {
        "loss_rpn_cls": 0.0,
        "loss_rpn_reg": 0.0,
        "loss_fast_rcnn_cls": 0.0,
        "loss_fast_rcnn_reg": 0.0,
        "loss_total": 0.0
    }
    
    total_batches = len(dataloader)
    start_time = time.time()
    
    for idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients to avoid exploding gradient issues
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Track statistics
        running_losses["loss_total"] += total_loss.item()
        for k, v in loss_dict.items():
            running_losses[k] += v.item()
            
        if (idx + 1) % print_freq == 0 or (idx + 1) == total_batches:
            avg_losses = {k: v / (idx + 1) for k, v in running_losses.items()}
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}] Batch [{idx+1}/{total_batches}] | "
                f"Total Loss: {avg_losses['loss_total']:.4f} | "
                f"RPN Cls: {avg_losses['loss_rpn_cls']:.4f}, Reg: {avg_losses['loss_rpn_reg']:.4f} | "
                f"Fast R-CNN Cls: {avg_losses['loss_fast_rcnn_cls']:.4f}, Reg: {avg_losses['loss_fast_rcnn_reg']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
            
    epoch_losses = {k: v / total_batches for k, v in running_losses.items()}
    return epoch_losses


# ======================================================================
# 13. Evaluation Function
# ======================================================================
def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using all-point integration.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Calculate precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
        
    # Integrate Precision-Recall curve
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])
    return float(ap)

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, config: Config) -> float:
    """
    Evaluates model on validation loader and calculates custom mAP@50.
    """
    model.eval()
    
    # Structures to log all predictions and GT targets
    all_gts = {c: [] for c in range(1, config.num_classes + 1)}
    total_gts_count = {c: 0 for c in range(1, config.num_classes + 1)}
    all_preds = {c: [] for c in range(1, config.num_classes + 1)}
    
    print("\n--- Evaluating Model Performance ---")
    for img_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        preds = model(images)
        
        # Accumulate GT targets
        for i, target in enumerate(targets):
            global_img_idx = img_idx * dataloader.batch_size + i
            gt_boxes = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy()
            
            for box, label in zip(gt_boxes, gt_labels):
                if label in all_gts:
                    all_gts[label].append({
                        "img_idx": global_img_idx,
                        "box": box,
                        "matched": False
                    })
                    total_gts_count[label] += 1
                    
        # Accumulate Predictions
        for i, pred in enumerate(preds):
            global_img_idx = img_idx * dataloader.batch_size + i
            pred_boxes = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if label in all_preds:
                    all_preds[label].append({
                        "img_idx": global_img_idx,
                        "box": box,
                        "score": float(score)
                    })
                    
    # Calculate Class-wise AP@50
    aps = []
    print("Class-wise AP@50 Results:")
    
    for c in range(1, config.num_classes + 1):
        c_gts = all_gts[c]
        c_preds = all_preds[c]
        total_gt = total_gts_count[c]
        
        if total_gt == 0:
            print(f"  Class {config.class_names[c-1]}: No ground truth boxes in dataset.")
            continue
            
        if len(c_preds) == 0:
            print(f"  Class {config.class_names[c-1]}: AP@50 = 0.0000 (No predictions)")
            aps.append(0.0)
            continue
            
        # Sort predictions by score descending
        c_preds = sorted(c_preds, key=lambda x: x["score"], reverse=True)
        
        tps = np.zeros(len(c_preds))
        fps = np.zeros(len(c_preds))
        
        # Group GTs by image for fast queries
        gts_by_img = {}
        for gt in c_gts:
            img = gt["img_idx"]
            if img not in gts_by_img:
                gts_by_img[img] = []
            gts_by_img[img].append(gt)
            
        for p_idx, pred in enumerate(c_preds):
            p_img = pred["img_idx"]
            p_box = pred["box"]
            
            if p_img not in gts_by_img:
                fps[p_idx] = 1.0
                continue
                
            img_gts = gts_by_img[p_img]
            best_iou = -1.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(img_gts):
                g_box = gt["box"]
                
                # Intersection
                ixmin = max(p_box[0], g_box[0])
                iymin = max(p_box[1], g_box[1])
                ixmax = min(p_box[2], g_box[2])
                iymax = min(p_box[3], g_box[3])
                
                iw = max(0.0, ixmax - ixmin)
                ih = max(0.0, iymax - iymin)
                
                inters = iw * ih
                
                # Union
                uni = (
                    (p_box[2] - p_box[0]) * (p_box[3] - p_box[1]) +
                    (g_box[2] - g_box[0]) * (g_box[3] - g_box[1]) -
                    inters
                )
                
                iou = inters / max(uni, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= 0.5:
                if not img_gts[best_gt_idx]["matched"]:
                    tps[p_idx] = 1.0
                    img_gts[best_gt_idx]["matched"] = True
                else:
                    fps[p_idx] = 1.0
            else:
                fps[p_idx] = 1.0
                
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        
        recalls = cum_tp / total_gt
        precisions = cum_tp / (cum_tp + cum_fp)
        
        ap = compute_ap(recalls, precisions)
        print(f"  - {config.class_names[c-1]:<15}: AP@50 = {ap:.4f} | Predictions: {len(c_preds)}, GTs: {total_gt}")
        aps.append(ap)
        
    mAP = sum(aps) / len(aps) if len(aps) > 0 else 0.0
    print(f"\nFinal Validation Metric -> mAP@50 = {mAP:.4f}\n")
    return mAP


# ======================================================================
# 14. Inference Function
# ======================================================================
def inference(model: nn.Module, image_path: str, save_output_path: str, config: Config):
    """
    Runs model inference on a single image and writes predictions to output path.
    """
    model.eval()
    device = torch.device(config.device)
    
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    transform = T.Compose([
        T.Resize(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(orig_img).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        preds = model(img_tensor)
    elapsed = time.time() - start_time
    
    pred = preds[0]
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()
    
    # Filter with higher score threshold for clean demo visualization
    keep = torch.where(scores >= config.final_inference_score_thresh)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Scale boxes back to original PIL Image coordinates
    scale_x = orig_w / config.image_size[0]
    scale_y = orig_h / config.image_size[1]
    
    boxes[:, 0] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 2] *= scale_x
    boxes[:, 3] *= scale_y
    
    draw = ImageDraw.Draw(orig_img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    print(f"Inference run complete. Processing time: {elapsed*1000:.1f}ms. Found {len(boxes)} boxes above {config.final_inference_score_thresh*100}%.")
    
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3", "#33FFF3"]
    
    for box, score, label in zip(boxes, scores, labels):
        class_idx = int(label.item()) - 1
        class_name = config.class_names[class_idx]
        color = colors[class_idx % len(colors)]
        
        box_coords = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        draw.rectangle(box_coords, outline=color, width=3)
        
        text = f"{class_name}: {score.item()*100:.1f}%"
        text_bbox = draw.textbbox((box_coords[0], box_coords[1] - 18), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box_coords[0], box_coords[1] - 18), text, fill="black", font=font)
        
    os.makedirs(os.path.dirname(save_output_path), exist_ok=True)
    orig_img.save(save_output_path)
    print(f"Visualization saved to: {save_output_path}")


# ======================================================================
# 15. main()
# ======================================================================
def main():
    print("======================================================================")
    print("      CUSTOM FASTER R-CNN TRAINING - GARBAGE CLASSIFICATION           ")
    print("======================================================================")
    
    # Initialize Config
    config = Config()
    device = torch.device(config.device)
    print(f"Using execution device: {device}")
    
    # 1. Datasets & Dataloaders
    print("Initializing datasets...")
    train_dataset = GarbageDataset(
        images_dir=config.train_images,
        labels_dir=config.train_labels,
        image_size=config.image_size
    )
    
    val_dataset = GarbageDataset(
        images_dir=config.val_images,
        labels_dir=config.val_labels,
        image_size=config.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Loaded {len(train_dataset)} training samples.")
    print(f"Loaded {len(val_dataset)} validation samples.")
    
    # 2. Build Faster R-CNN Model
    print("Building Faster R-CNN model...")
    model = FasterRCNN(config=config)
    model = model.to(device)
    
    # 3. Setup Optimizer and Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=config.lr, 
        momentum=config.momentum, 
        weight_decay=config.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5, 
        gamma=0.1
    )
    
    # Checkpoints directory
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    
    best_mAP = 0.0
    
    # 4. Training Loop
    print("\nStarting training loop...")
    for epoch in range(1, config.epochs + 1):
        print(f"\n--- Epoch {epoch}/{config.epochs} (Learning Rate: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        # Train one epoch
        epoch_losses = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Step LR scheduler
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_mAP = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config
        )
        
        # Save best model checkpoint
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save(model.state_dict(), config.save_path)
            print(f"New best mAP achieved! Checkpoint saved to: {config.save_path}")
            
    print(f"\nTraining completed! Best Validation mAP@50: {best_mAP:.4f}")
    
    # 5. Run Demo Inference on Test Samples
    test_img_dir = config.test_images
    if os.path.exists(test_img_dir):
        test_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(test_files) > 0:
            print("\nLoading best model checkpoint for test inference...")
            if os.path.exists(config.save_path):
                model.load_state_dict(torch.load(config.save_path))
            else:
                print("Checkpoint file not found. Running inference with current model states.")
                
            demo_count = min(3, len(test_files))
            print(f"Running inference on {demo_count} sample test images...")
            for idx, test_file in enumerate(test_files[:demo_count]):
                test_img_path = os.path.join(test_img_dir, test_file)
                output_path = f"inference_outputs/prediction_{idx}.jpg"
                inference(model, test_img_path, output_path, config)
                
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
