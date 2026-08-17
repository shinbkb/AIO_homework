import os
import tarfile
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ==========================================
# 1. THIẾT LẬP DANH SÁCH CLASS PASCAL VOC
# ==========================================
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# ==========================================
# 2. LỚP DỮ LIỆU VOCDATASET
# ==========================================
class VOCDataset(Dataset):
    def __init__(self, root_dir, S=7, B=2, C=20, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        
        self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
        self.image_ids = [f.split('.')[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        xml_path = os.path.join(self.ann_dir, f"{img_id}.xml")

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[cls_name]
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) / orig_w
            ymin = float(bndbox.find("ymin").text) / orig_h
            xmax = float(bndbox.find("xmax").text) / orig_w
            ymax = float(bndbox.find("ymax").text) / orig_h
            
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            
            boxes.append([x_center, y_center, w, h])
            labels.append(label)

        if self.transform:
            image = self.transform(image)

        target = torch.zeros((self.S, self.S, self.C + 5))
        
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            
            grid_i = int(self.S * y)
            grid_j = int(self.S * x)
            
            x_cell = self.S * x - grid_j
            y_cell = self.S * y - grid_i
            
            if target[grid_i, grid_j, self.C + 4] == 0:
                target[grid_i, grid_j, label] = 1.0
                target[grid_i, grid_j, self.C:self.C + 4] = torch.tensor([x_cell, y_cell, w, h])
                target[grid_i, grid_j, self.C + 4] = 1.0

        return image, target


# ==========================================
# 3. MODULAR BACKBONE & YOLOV1 MODEL
# ==========================================
class ModularBackbone(nn.Module):
    """
    Modular Feature Extractor hỗ trợ 6 kiến trúc:
    - custom_cnn (YOLOv1 gốc)
    - vgg16
    - resnet18
    - resnet50
    - efficientnet_b0
    - mobilenet_v2
    """
    def __init__(self, backbone_name='custom_cnn', pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == 'custom_cnn':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(192, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, 2),
            )
            self.out_channels = 1024

        elif self.backbone_name == 'vgg16':
            try:
                vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                vgg = models.vgg16(pretrained=pretrained)
            self.features = vgg.features
            self.out_channels = 512

        elif self.backbone_name == 'resnet18':
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                resnet = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.out_channels = 512

        elif self.backbone_name == 'resnet50':
            try:
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                resnet = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.out_channels = 2048

        elif self.backbone_name == 'efficientnet_b0':
            try:
                effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                effnet = models.efficientnet_b0(pretrained=pretrained)
            self.features = effnet.features
            self.out_channels = 1280

        elif self.backbone_name == 'mobilenet_v2':
            try:
                mbnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            except AttributeError:
                mbnet = models.mobilenet_v2(pretrained=pretrained)
            self.features = mbnet.features
            self.out_channels = 1280
            
        else:
            raise ValueError(f"Backbone '{backbone_name}' không hợp lệ!")

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        
        self.backbone = ModularBackbone(backbone_name=backbone_name, pretrained=pretrained)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels * S * S, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (B * 5 + C))
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.view(-1, self.S, self.S, self.B * 5 + self.C)


# ==========================================
# 4. HÀM IOU VÀ YOLO LOSS FUNCTION
# ==========================================
def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, target):
        iou_b1 = intersection_over_union(predictions[..., 20:24], target[..., 20:24])
        iou_b2 = intersection_over_union(predictions[..., 25:29], target[..., 20:24])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        best_box = torch.argmax(ious, dim=0)
        exists_box = target[..., 24:25]

        # 1. Box Coordinate Loss (Tạo tensor mới tránh lỗi In-place operation)
        box_preds = (
            best_box * predictions[..., 25:29] + (1 - best_box) * predictions[..., 20:24]
        )
        box_targs = target[..., 20:24]

        pred_xy = box_preds[..., 0:2]
        pred_wh = torch.sign(box_preds[..., 2:4]) * torch.sqrt(
            torch.abs(box_preds[..., 2:4]) + 1e-6
        )
        box_predictions_transformed = torch.cat([pred_xy, pred_wh], dim=-1)

        targ_xy = box_targs[..., 0:2]
        targ_wh = torch.sqrt(torch.abs(box_targs[..., 2:4]) + 1e-6)
        box_targets_transformed = torch.cat([targ_xy, targ_wh], dim=-1)

        loss_coord = self.mse(
            torch.flatten(exists_box * box_predictions_transformed, end_dim=-2),
            torch.flatten(exists_box * box_targets_transformed, end_dim=-2),
        )

        # 2. Object Loss
        pred_conf = (
            best_box * predictions[..., 29:30] + (1 - best_box) * predictions[..., 24:25]
        )
        loss_obj = self.mse(
            torch.flatten(exists_box * pred_conf),
            torch.flatten(exists_box * target[..., 24:25]),
        )

        # 3. No Object Loss
        loss_noobj = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 24:25], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 24:25], start_dim=1),
        ) + self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 29:30], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 24:25], start_dim=1),
        )

        # 4. Class Loss
        loss_class = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        total_loss = (
            self.lambda_coord * loss_coord
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_class
        )

        return total_loss


# ==========================================
# 5. VÒNG LẶP HUẤN LUYỆN & MAIN
# ==========================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            print(f"  Batch [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đang sử dụng: {device}")

    transform = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
    ])

    voc_dir = "/home/shin-bkb/code/AIO_homework/yolov1/data/VOCdevkit/VOC2012"
    dataset = VOCDataset(root_dir=voc_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    print(f"Tổng số ảnh trong dataset: {len(dataset)}")

    backbones = ['custom_cnn', 'vgg16', 'resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']

    print("\n=== THỬ NGHIỆM TẤT CẢ BACKBONE ===")
    for bb in backbones:
        print(f"\n---> Đang thử nghiệm YOLOv1 với Backbone: {bb}")
        model = YOLOv1(backbone_name=bb, pretrained=True).to(device)
        criterion = YOLOLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"✅ Hoàn thành 1 Epoch | Backbone: {bb} | Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
