import os
import glob
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"Using device: {device}")


BASE_DIR = Path(__file__).resolve().parent
ZIP_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
ZIP_PATH = BASE_DIR / "mini_speech_commands.zip"
EXTRACT_DIR = BASE_DIR / "dataset_speech_commands"
TARGET_CLASSES = ["go", "stop", "yes", "no"]
MAX_SAMPLES_PER_CLASS = 150


class AudioCNN1D(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def progress(iterable, desc):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False)
    return iterable


def load_dataset():
    if not ZIP_PATH.exists():
        print("Đang tải dữ liệu Speech Commands từ internet...")
        urllib.request.urlretrieve(ZIP_URL, str(ZIP_PATH))

    if not EXTRACT_DIR.exists():
        print("Đang giải nén bộ dữ liệu...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(str(EXTRACT_DIR))

    dataset_dir = EXTRACT_DIR / "mini_speech_commands"
    X_audio = []
    y_labels = []

    for label, class_name in enumerate(TARGET_CLASSES):
        class_dir = dataset_dir / class_name
        wav_files = sorted(glob.glob(str(class_dir / "*.wav")))

        for wav_path in wav_files[:MAX_SAMPLES_PER_CLASS]:
            signal, _ = librosa.load(wav_path, sr=16000, duration=1.0)

            if len(signal) < 16000:
                signal = np.pad(signal, (0, 16000 - len(signal)))
            else:
                signal = signal[:16000]

            X_audio.append(signal)
            y_labels.append(label)

    print(f"Tổng số mẫu âm thanh đã nạp: {len(X_audio)}")
    print(f"Độ dài mỗi mẫu âm thanh: {len(X_audio[0]) if X_audio else 0} samples")
    return np.array(X_audio, dtype=np.float32), np.array(y_labels, dtype=np.int64)


def build_dataloaders(X, y):
    X = np.expand_dims(X, axis=1)  # (N, 1, 16000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Test: {X_test.shape}")
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=25, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in progress(train_loader, f"Epoch {epoch + 1}/{epochs} [train]"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_batch, y_batch in progress(test_loader, f"Epoch {epoch + 1}/{epochs} [test]"):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                test_loss_sum += loss.item() * X_batch.size(0)
                predictions = torch.argmax(outputs, dim=1)
                test_correct += (predictions == y_batch).sum().item()
                test_total += y_batch.size(0)

        test_loss = test_loss_sum / test_total
        test_acc = test_correct / test_total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

    return train_losses, test_losses, train_accs, test_accs


def plot_learning_curves(train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "learning_curve.png", dpi=150)
    plt.show()


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(predictions.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=TARGET_CLASSES))
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
    )
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thật")
    plt.title("Confusion Matrix")
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.show()


def main():
    X, y = load_dataset()
    train_loader, test_loader = build_dataloaders(X, y)

    model = AudioCNN1D(num_classes=4).to(device)
    print(model)

    train_losses, test_losses, train_accs, test_accs = train_model(
        model, train_loader, test_loader, epochs=25, lr=1e-3
    )

    plot_learning_curves(train_losses, test_losses, train_accs, test_accs)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
