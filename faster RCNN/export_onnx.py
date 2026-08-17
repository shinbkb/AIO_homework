import os
import argparse
import torch
import torch.nn as nn
from infer_rcnn import FasterRCNN, Config

class FasterRCNN_BackboneRPN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.rpn = model.rpn

    def forward(self, x):
        features = self.backbone(x)
        cls_logits, bbox_deltas, anchors, _ = self.rpn(features)
        return features, cls_logits, bbox_deltas

class FasterRCNN_Heads(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.classification_head = model.classification_head
        self.bbox_head = model.bbox_head

    def forward(self, roi_features):
        cls_logits, shared_features = self.classification_head(roi_features)
        bbox_deltas = self.bbox_head(shared_features)
        return cls_logits, bbox_deltas

def export_to_onnx(weights_path: str, output_dir: str = "."):
    config = Config()
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

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Backbone + RPN
    print("📦 1/2. Đang xuất ONNX cho Backbone ResNet50 + RPN...")
    model_rpn = FasterRCNN_BackboneRPN(model)
    dummy_img = torch.randn(1, 3, 416, 416)
    rpn_onnx_path = os.path.join(output_dir, "faster_rcnn_backbone_rpn.onnx")
    
    torch.onnx.export(
        model_rpn,
        dummy_img,
        rpn_onnx_path,
        dynamo=False,
        opset_version=14,
        input_names=["input"],
        output_names=["features", "rpn_cls_logits", "rpn_bbox_deltas"]
    )
    print(f"✅ Đã lưu: {rpn_onnx_path}")

    # 2. Export Classification & Bbox Heads
    print("📦 2/2. Đang xuất ONNX cho Classification & Box Heads...")
    model_heads = FasterRCNN_Heads(model)
    dummy_roi = torch.randn(10, 1024, 7, 7)
    heads_onnx_path = os.path.join(output_dir, "faster_rcnn_heads.onnx")

    torch.onnx.export(
        model_heads,
        dummy_roi,
        heads_onnx_path,
        dynamo=False,
        opset_version=14,
        input_names=["roi_features"],
        output_names=["cls_logits", "bbox_deltas"],
        dynamic_axes={
            "roi_features": {0: "num_rois"},
            "cls_logits": {0: "num_rois"},
            "bbox_deltas": {0: "num_rois"}
        }
    )
    print(f"✅ Đã lưu: {heads_onnx_path}")
    print("\n🎉 Xuất ONNX thành công! Hãy copy 2 file .onnx này lên Raspberry Pi để chạy tệp suy luận siêu tốc.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="faster_rcnn_garbage.pth", help="File trọng số .pth")
    parser.add_argument("--outdir", type=str, default=".", help="Thư mục lưu file .onnx")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.outdir)
