import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import threading
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.ops import roi_align, nms

try:
    from flask import Flask, Response, render_template_string
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

class Config:
    image_size = (416, 416)
    class_names = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    num_classes = len(class_names)
    
    backbone_out_channels = 1024
    anchor_scales = [64, 128, 256]
    anchor_ratios = [0.5, 1.0, 2.0]
    
    rpn_pre_nms_top_n_test = 6000
    rpn_post_nms_top_n_test = 300
    rpn_nms_thresh = 0.7
    
    roi_output_size = (7, 7)
    roi_spatial_scale = 1.0 / 16.0
    
    final_score_thresh = 0.3
    final_nms_thresh = 0.3
    device = "cpu"

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if hasattr(torchvision.models, 'ResNet50_Weights'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
        else:
            resnet = torchvision.models.resnet50(pretrained=pretrained)
            
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def generate_anchors(feature_h: int, feature_w: int, stride: int, 
                     scales: list, ratios: list, device: torch.device) -> torch.Tensor:
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            h = scale / math.sqrt(ratio)
            w = scale * math.sqrt(ratio)
            base_anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
            
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)
    
    shift_x = torch.arange(0, feature_w, dtype=torch.float32, device=device) * stride + stride / 2.0
    shift_y = torch.arange(0, feature_h, dtype=torch.float32, device=device) * stride + stride / 2.0
    
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    
    shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
    all_anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
    return all_anchors.reshape(-1, 4)

def box_decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    x_a = anchors[:, 0] + 0.5 * w_a
    y_a = anchors[:, 1] + 0.5 * h_a
    
    tx = deltas[:, 0]
    ty = deltas[:, 1]
    tw = deltas[:, 2]
    th = deltas[:, 3]
    
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
    def __init__(self, in_channels: int, num_anchors: int, config: Config):
        super().__init__()
        self.config = config
        self.num_anchors = num_anchors
        
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, kernel_size=1)

    def forward(self, features: torch.Tensor):
        batch_size, _, feat_h, feat_w = features.shape
        device = features.device
        
        t = self.relu(self.conv(features))
        cls_logits = self.cls_layer(t).permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        bbox_deltas = self.reg_layer(t).permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        anchors = generate_anchors(
            feature_h=feat_h, feature_w=feat_w, stride=16,
            scales=self.config.anchor_scales, ratios=self.config.anchor_ratios, device=device
        )
        return cls_logits, bbox_deltas, anchors

class ProposalGenerator:
    def __init__(self, config: Config):
        self.config = config

    @torch.no_grad()
    def __call__(self, cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, 
                 anchors: torch.Tensor) -> list:
        batch_size = cls_logits.shape[0]
        pre_nms_top_n = self.config.rpn_pre_nms_top_n_test
        post_nms_top_n = self.config.rpn_post_nms_top_n_test
        
        scores = F.softmax(cls_logits, dim=-1)[:, :, 1]
        proposals = []
        for i in range(batch_size):
            img_scores = scores[i]
            img_deltas = bbox_deltas[i]
            decoded_boxes = box_decode(img_deltas, anchors)
            
            w_img, h_img = self.config.image_size
            decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
            decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
            decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
            decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
            
            ws = decoded_boxes[:, 2] - decoded_boxes[:, 0]
            hs = decoded_boxes[:, 3] - decoded_boxes[:, 1]
            keep = (ws >= 1.0) & (hs >= 1.0)
            
            img_scores = img_scores[keep]
            decoded_boxes = decoded_boxes[keep]
            
            if len(decoded_boxes) == 0:
                proposals.append(anchors[:10].clone())
                continue
                
            k = min(len(decoded_boxes), pre_nms_top_n)
            topk_scores, topk_idx = torch.topk(img_scores, k)
            topk_boxes = decoded_boxes[topk_idx]
            
            keep_nms = nms(topk_boxes, topk_scores, self.config.rpn_nms_thresh)[:post_nms_top_n]
            proposals.append(topk_boxes[keep_nms])
            
        return proposals

class RoIHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, features: torch.Tensor, proposals: list) -> torch.Tensor:
        return roi_align(
            features, proposals, output_size=self.config.roi_output_size, spatial_scale=self.config.roi_spatial_scale
        )

class ClassificationHead(nn.Module):
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

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], -1)
        x = self.dropout1(self.relu1(self.fc1(x)))
        shared_features = self.dropout2(self.relu2(self.fc2(x)))
        cls_logits = self.cls_score(shared_features)
        return cls_logits, shared_features

class BoundingBoxHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.bbox_pred = nn.Linear(in_features, (num_classes + 1) * 4)

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        return self.bbox_pred(shared_features)

class FasterRCNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.backbone = ResNetBackbone(pretrained=False)
        num_anchors = len(config.anchor_scales) * len(config.anchor_ratios)
        self.rpn = RegionProposalNetwork(in_channels=config.backbone_out_channels, num_anchors=num_anchors, config=config)
        self.proposal_generator = ProposalGenerator(config=config)
        self.roi_head = RoIHead(config=config)
        self.classification_head = ClassificationHead(in_channels=config.backbone_out_channels, num_classes=config.num_classes, config=config)
        self.bbox_head = BoundingBoxHead(in_features=1024, num_classes=config.num_classes)

    def forward(self, images: torch.Tensor) -> list:
        batch_size = images.shape[0]
        device = images.device
        
        features = self.backbone(images)
        rpn_cls_logits, rpn_bbox_deltas, anchors = self.rpn(features)
        proposals = self.proposal_generator(rpn_cls_logits, rpn_bbox_deltas, anchors)
        
        roi_features = self.roi_head(features, proposals)
        cls_logits, shared_features = self.classification_head(roi_features)
        bbox_deltas = self.bbox_head(shared_features)
        
        cls_probs = F.softmax(cls_logits, dim=-1)
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
            
            for c in range(1, self.config.num_classes + 1):
                class_scores = img_cls_probs[:, c]
                keep_idx = torch.where(class_scores > self.config.final_score_thresh)[0]
                if len(keep_idx) == 0:
                    continue
                    
                keep_scores = class_scores[keep_idx]
                keep_proposals = img_proposals[keep_idx]
                keep_deltas = img_bbox_deltas[keep_idx, 4*c : 4*c+4]
                
                decoded_boxes = box_decode(keep_deltas, keep_proposals)
                decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
                decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
                decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
                decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
                
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

def draw_predictions_on_frame(frame, pred, config, score_thresh):
    if pred is None:
        return frame
        
    boxes = pred["boxes"].cpu()
    scores = pred["scores"].cpu()
    labels = pred["labels"].cpu()

    keep = torch.where(scores >= score_thresh)[0]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    orig_h, orig_w = frame.shape[:2]
    scale_x = orig_w / float(config.image_size[0])
    scale_y = orig_h / float(config.image_size[1])

    colors = [
        (51, 87, 255),   # BIODEGRADABLE
        (87, 255, 51),   # CARDBOARD
        (255, 87, 51),   # GLASS
        (51, 255, 243),  # METAL
        (243, 51, 255),  # PAPER
        (243, 255, 51)   # PLASTIC
    ]

    for box, score, label in zip(boxes, scores, labels):
        class_idx = int(label.item()) - 1
        class_name = config.class_names[class_idx]
        color = colors[class_idx % len(colors)]

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score.item()*100:.0f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + text_w, max(0, y1)), color, -1)
        cv2.putText(frame, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

# Multi-threading Shared Variables for Smooth 30 FPS Stream
latest_frame = None
latest_prediction = None
is_inferencing = False
lock = threading.Lock()

def inference_worker(model, config, transform, device):
    global latest_frame, latest_prediction, is_inferencing
    while True:
        with lock:
            frame_to_process = latest_frame.copy() if latest_frame is not None else None

        if frame_to_process is not None:
            is_inferencing = True
            # Resize image down to 320x240 before model processing to save CPU
            h_orig, w_orig = frame_to_process.shape[:2]
            if w_orig > 640:
                frame_to_process = cv2.resize(frame_to_process, (640, 480))

            frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(img_tensor)

            with lock:
                latest_prediction = preds[0]
            is_inferencing = False
            
        time.sleep(0.1)

def start_camera_stream(weights_path: str, camera_input: str = "0", port: int = 5000, score_thresh: float = 0.3):
    global latest_frame, latest_prediction
    config = Config()
    config.final_score_thresh = score_thresh
    device = torch.device("cpu")
    
    print(f"⏳ Đang nạp mô hình từ {weights_path}...")
    model = FasterRCNN(config=config)
    
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        cam_source = int(camera_input)
    except ValueError:
        cam_source = camera_input

    print(f"📷 Đang mở camera nguồn: {cam_source}...")
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"❌ Không thể kết nối với Camera nguồn '{cam_source}'. Vui lòng kiểm tra lại địa chỉ URL hoặc kết nối camera.")
        return

    # Start background AI Inference thread
    ai_thread = threading.Thread(target=inference_worker, args=(model, config, transform, device), daemon=True)
    ai_thread.start()

    if HAS_FLASK:
        app = Flask(__name__)

        HTML_PAGE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Raspberry Pi Garbage Detection Live Camera</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #1e1e2e; color: white; margin: 0; padding: 20px; }
                h1 { color: #89b4fa; }
                .container { display: inline-block; border: 4px solid #89b4fa; border-radius: 12px; overflow: hidden; background: #000; }
                img { width: 640px; height: 480px; }
                .info { margin-top: 15px; color: #a6adc8; }
            </style>
        </head>
        <body>
            <h1>🎥 Live Camera Garbage Detection (Fast Multi-threaded)</h1>
            <div class="container">
                <img src="/video_feed" />
            </div>
            <div class="info">
                <p>⚡ Luồng Video được tách riêng (30 FPS mượt mà). Bounding Box cập nhật ngầm theo mô hình.</p>
            </div>
        </body>
        </html>
        """

        def generate_frames():
            global latest_frame, latest_prediction
            last_frame_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                with lock:
                    latest_frame = frame.copy()
                    pred = latest_prediction

                now = time.time()
                fps = 1.0 / (now - last_frame_time + 1e-6)
                last_frame_time = now

                # Draw bounding boxes from latest prediction
                frame_annotated = draw_predictions_on_frame(frame, pred, config, score_thresh)
                cv2.putText(frame_annotated, f"Stream FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_inferencing:
                    cv2.putText(frame_annotated, "AI Processing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                _, buffer = cv2.imencode('.jpg', frame_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.03) # Smooth 30 FPS playback

        @app.route('/')
        def index():
            return render_template_string(HTML_PAGE)

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        print(f"🌐 Server Web Camera đã mở tại: http://0.0.0.0:{port}")
        print(f"👉 Truy cập trên trình duyệt laptop: http://raspberrypi.local:{port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="faster_rcnn_garbage.pth", help="Đường dẫn file .pth")
    parser.add_argument("--camera", type=str, default="0", help="ID Camera (0) hoặc URL IP Camera từ điện thoại")
    parser.add_argument("--port", type=int, default=5000, help="Cổng Web Stream (mặc định 5000)")
    parser.add_argument("--thresh", type=float, default=0.3, help="Ngưỡng tự tin score_thresh (mặc định 0.3)")
    args = parser.parse_args()

    start_camera_stream(args.weights, camera_input=args.camera, port=args.port, score_thresh=args.thresh)
