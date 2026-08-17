import os
import sys
import time
import math
import argparse
import numpy as np
import cv2
import threading
from PIL import Image

import onnxruntime as ort
import torch
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
    
    anchor_scales = [64, 128, 256]
    anchor_ratios = [0.5, 1.0, 2.0]
    
    rpn_pre_nms_top_n_test = 3000
    rpn_post_nms_top_n_test = 200
    rpn_nms_thresh = 0.7
    
    roi_output_size = (7, 7)
    roi_spatial_scale = 1.0 / 16.0
    
    final_score_thresh = 0.3
    final_nms_thresh = 0.3

def generate_anchors(feature_h: int, feature_w: int, stride: int, 
                     scales: list, ratios: list) -> torch.Tensor:
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            h = scale / math.sqrt(ratio)
            w = scale * math.sqrt(ratio)
            base_anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
            
    base_anchors = torch.tensor(base_anchors, dtype=torch.float32)
    
    shift_x = torch.arange(0, feature_w, dtype=torch.float32) * stride + stride / 2.0
    shift_y = torch.arange(0, feature_h, dtype=torch.float32) * stride + stride / 2.0
    
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

def get_proposals_from_rpn(cls_logits: torch.Tensor, bbox_deltas: torch.Tensor, 
                           anchors: torch.Tensor, config: Config) -> torch.Tensor:
    scores = torch.softmax(cls_logits, dim=-1)[0, :, 1]
    deltas = bbox_deltas[0]
    decoded_boxes = box_decode(deltas, anchors)
    
    w_img, h_img = config.image_size
    decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
    decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
    decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
    decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
    
    ws = decoded_boxes[:, 2] - decoded_boxes[:, 0]
    hs = decoded_boxes[:, 3] - decoded_boxes[:, 1]
    keep = (ws >= 1.0) & (hs >= 1.0)
    
    scores = scores[keep]
    decoded_boxes = decoded_boxes[keep]
    
    if len(decoded_boxes) == 0:
        return anchors[:10].clone()
        
    k = min(len(decoded_boxes), config.rpn_pre_nms_top_n_test)
    topk_scores, topk_idx = torch.topk(scores, k)
    topk_boxes = decoded_boxes[topk_idx]
    
    keep_nms = nms(topk_boxes, topk_scores, config.rpn_nms_thresh)[:config.rpn_post_nms_top_n_test]
    return topk_boxes[keep_nms]

class FasterRCNN_ONNXEngine:
    def __init__(self, rpn_onnx_path: str, heads_onnx_path: str, config: Config):
        self.config = config
        
        # Configure ONNX Runtime session for fast CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print("⚡ Đang nạp ONNX Engine cho RPN & Backbone...")
        self.sess_rpn = ort.InferenceSession(rpn_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        print("⚡ Đang nạp ONNX Engine cho Heads...")
        self.sess_heads = ort.InferenceSession(heads_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])

    def predict(self, img_tensor: torch.Tensor):
        img_np = img_tensor.numpy()
        
        # 1. Run ONNX Backbone + RPN
        features_np, rpn_cls_np, rpn_reg_np = self.sess_rpn.run(None, {'input': img_np})
        
        features_t = torch.from_numpy(features_np)
        rpn_cls_t = torch.from_numpy(rpn_cls_np)
        rpn_reg_t = torch.from_numpy(rpn_reg_np)
        
        # Generate Anchors
        feat_h, feat_w = features_np.shape[2], features_np.shape[3]
        anchors = generate_anchors(feat_h, feat_w, 16, self.config.anchor_scales, self.config.anchor_ratios)
        
        # RPN Proposal Generation
        proposals = get_proposals_from_rpn(rpn_cls_t, rpn_reg_t, anchors, self.config)
        
        # RoI Align
        roi_feats = roi_align(features_t, [proposals], output_size=self.config.roi_output_size, spatial_scale=self.config.roi_spatial_scale)
        
        # 2. Run ONNX Classification & Bbox Heads
        cls_logits_np, bbox_deltas_np = self.sess_heads.run(None, {'roi_features': roi_feats.numpy()})
        
        cls_probs = torch.softmax(torch.from_numpy(cls_logits_np), dim=-1)
        bbox_deltas = torch.from_numpy(bbox_deltas_np)
        
        img_final_boxes = []
        img_final_scores = []
        img_final_labels = []
        w_img, h_img = self.config.image_size
        
        for c in range(1, self.config.num_classes + 1):
            class_scores = cls_probs[:, c]
            keep_idx = torch.where(class_scores > self.config.final_score_thresh)[0]
            if len(keep_idx) == 0:
                continue
                
            keep_scores = class_scores[keep_idx]
            keep_proposals = proposals[keep_idx]
            keep_deltas = bbox_deltas[keep_idx, 4*c : 4*c+4]
            
            decoded_boxes = box_decode(keep_deltas, keep_proposals)
            decoded_boxes[:, 0] = torch.clamp(decoded_boxes[:, 0], min=0.0, max=w_img)
            decoded_boxes[:, 1] = torch.clamp(decoded_boxes[:, 1], min=0.0, max=h_img)
            decoded_boxes[:, 2] = torch.clamp(decoded_boxes[:, 2], min=0.0, max=w_img)
            decoded_boxes[:, 3] = torch.clamp(decoded_boxes[:, 3], min=0.0, max=h_img)
            
            keep_nms = nms(decoded_boxes, keep_scores, self.config.final_nms_thresh)
            img_final_boxes.append(decoded_boxes[keep_nms])
            img_final_scores.append(keep_scores[keep_nms])
            img_final_labels.append(torch.full((len(keep_nms),), c, dtype=torch.int64))
            
        if len(img_final_boxes) > 0:
            return {
                "boxes": torch.cat(img_final_boxes, dim=0),
                "scores": torch.cat(img_final_scores, dim=0),
                "labels": torch.cat(img_final_labels, dim=0)
            }
        else:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }

def draw_predictions_on_frame(frame, pred, config, score_thresh):
    if pred is None:
        return frame
        
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]

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

latest_frame = None
latest_prediction = None
is_inferencing = False
lock = threading.Lock()

def inference_worker(engine, config, transform):
    global latest_frame, latest_prediction, is_inferencing
    while True:
        with lock:
            frame_to_process = latest_frame.copy() if latest_frame is not None else None

        if frame_to_process is not None:
            is_inferencing = True
            h_orig, w_orig = frame_to_process.shape[:2]
            if w_orig > 640:
                frame_to_process = cv2.resize(frame_to_process, (640, 480))

            frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tensor = transform(img_pil).unsqueeze(0)

            t0 = time.time()
            pred = engine.predict(img_tensor)
            elapsed = (time.time() - t0) * 1000.0

            with lock:
                latest_prediction = pred
            is_inferencing = False
            
        time.sleep(0.05)

def start_camera_stream(rpn_onnx: str, heads_onnx: str, camera_input: str = "0", port: int = 5000, score_thresh: float = 0.3):
    global latest_frame, latest_prediction
    config = Config()
    config.final_score_thresh = score_thresh
    
    engine = FasterRCNN_ONNXEngine(rpn_onnx, heads_onnx, config)

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
        print(f"❌ Không thể kết nối với Camera nguồn '{cam_source}'. Vui lòng kiểm tra kết nối.")
        return

    ai_thread = threading.Thread(target=inference_worker, args=(engine, config, transform), daemon=True)
    ai_thread.start()

    if HAS_FLASK:
        app = Flask(__name__)

        HTML_PAGE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Raspberry Pi ONNX Fast Camera Stream</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #181825; color: white; margin: 0; padding: 20px; }
                h1 { color: #a6e3a1; }
                .container { display: inline-block; border: 4px solid #a6e3a1; border-radius: 12px; overflow: hidden; background: #000; }
                img { width: 640px; height: 480px; }
                .info { margin-top: 15px; color: #bac2de; }
            </style>
        </head>
        <body>
            <h1>⚡ ONNX Runtime Live Camera Detection</h1>
            <div class="container">
                <img src="/video_feed" />
            </div>
            <div class="info">
                <p>🚀 Đã tăng tốc bằng ONNX Runtime C++ Engine trên Raspberry Pi!</p>
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

                frame_annotated = draw_predictions_on_frame(frame, pred, config, score_thresh)
                cv2.putText(frame_annotated, f"Stream FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if is_inferencing:
                    cv2.putText(frame_annotated, "ONNX AI Processing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                _, buffer = cv2.imencode('.jpg', frame_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.03)

        @app.route('/')
        def index():
            return render_template_string(HTML_PAGE)

        @app.route('/video_feed')
        def video_feed():
            return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        print(f"🌐 ONNX Web Server đang chạy tại: http://0.0.0.0:{port}")
        print(f"👉 Truy cập trên laptop: http://raspberrypi.local:{port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpn-onnx", type=str, default="faster_rcnn_backbone_rpn.onnx", help="File RPN .onnx")
    parser.add_argument("--heads-onnx", type=str, default="faster_rcnn_heads.onnx", help="File Heads .onnx")
    parser.add_argument("--camera", type=str, default="0", help="ID Camera hoặc URL IP Camera")
    parser.add_argument("--port", type=int, default=5000, help="Cổng Web Stream (mặc định 5000)")
    parser.add_argument("--thresh", type=float, default=0.3, help="Ngưỡng tự tin score_thresh (mặc định 0.3)")
    args = parser.parse_args()

    start_camera_stream(args.rpn_onnx, args.heads_onnx, camera_input=args.camera, port=args.port, score_thresh=args.thresh)
