
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm


from google.colab import files
files.upload()
os.makedirs("/root/.kaggle",exist_ok=True)
os.rename("kaggle.json","/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json",600)

!kaggle datasets download -d sujaykapadnis/emotion-recognition-dataset
!unzip -q emotion-recognition-dataset.zip -d /content/emotion_data
!ls /content/emotion_data  # kiểm tra thư mục

for root, dirs, files in os.walk("/content/emotion_data"):
    level = root.replace("/content/emotion_data", "").count(os.sep)
    if level < 3:
        print(" " * level * 2 + os.path.basename(root) + "/")

DATA_DIR = "/content/emotion_data/dataset"  
NUM_CLASSES = 6          # Ahegao, Angry, Happy, Neutral, Sad, Surprise
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
IMG_SIZE    = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Load dataset — tách 80% train, 20% val
full_dataset = datasets.ImageFolder(DATA_DIR)
n_total = len(full_dataset)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])
# Gán transform riêng cho mỗi phần
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform   = val_transform
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f"Train: {n_train} | Val: {n_val} | Classes: {full_dataset.classes}")

# Load VGG16 pretrained
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# ĐÓNG BĂNG tất cả layers trước
for param in model.features.parameters():
    param.requires_grad = False

# MỞ BĂNG Block 5 (từ layer thứ 24 trở đi trong features của VGG16)
for param in model.features[24:].parameters():
    param.requires_grad = True

# Thay thế classifier cho bài toán 6 classes
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
    nn.Linear(4096, 4096),        nn.ReLU(inplace=True), nn.Dropout(0.5),
    nn.Linear(4096, NUM_CLASSES),
)

model = model.to(DEVICE)

# Kiểm tra số tham số cần train
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

criterion = nn.CrossEntropyLoss()

# Tối ưu hóa cả classifier và các conv layer đã mở băng
optimizer = optim.Adam([
    {'params': model.features[24:].parameters(), 'lr': 1e-5}, # LR cực nhỏ cho features
    {'params': model.classifier.parameters(), 'lr': 5e-5}     # LR lớn hơn chút cho classifier
])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader,desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += imgs.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += imgs.size(0)
    return running_loss / total, correct / total

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
    elapsed = time.time() - t0

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    print(f"Epoch {epoch+1:02}/{NUM_EPOCHS} | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val  Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

epochs = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history["train_loss"], label="Train Loss")
axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
axes[0].set_title("Loss"); axes[0].legend()

axes[1].plot(epochs, history["train_acc"], label="Train Acc")
axes[1].plot(epochs, history["val_acc"],   label="Val Acc")
axes[1].set_title("Accuracy"); axes[1].legend()

plt.suptitle("VGG16 - Feature Extraction (Fine Tuning)")
plt.tight_layout()
plt.show()

print(f"\nFinal Val Accuracy: {history['val_acc'][-1]*100:.2f}%")