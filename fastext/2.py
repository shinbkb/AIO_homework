import os
import re
import zipfile

import numpy as np
import fasttext
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


TRAIN_ZIP = (
    "/content/drive/My Drive/AIO_Homework/dence representation/data/data_train.zip"
)
TEST_ZIP = (
    "/content/drive/My Drive/AIO_Homework/dence representation/data/data_test.zip"
)
DATASET_PATH = "/content/dataset/"

TRAIN_DIR = "/content/dataset/data_train/train"
VAL_DIR   = "/content/dataset/data_train/test"
TEST_DIR  = "/content/dataset/data_test/test"

# FastText pretrained
FT_MODEL_URL  = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz"
)
FT_MODEL_PATH = "cc.vi.300.bin"

# Siêu tham số MLP
EMBEDDING_DIM   = 300
HIDDEN_DIM      = 256
NUM_CLASSES     = 2
BATCH_SIZE      = 64
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 20
PATIENCE        = 5
BEST_MODEL_PATH = "best_mlp.pt"


def unzip(zip_path: str, extract_to: str) -> None:
    """Giải nén file zip vào thư mục chỉ định."""
    print(f"Đang giải nén {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Hoàn thành giải nén vào {extract_to}")


def load_corpus_with_labels(directory: str):
    """
    Đọc văn bản và nhãn từ cấu trúc thư mục:
      directory/neg/  -> nhãn 0
      directory/pos/  -> nhãn 1

    Returns:
        texts  (list[str]): Nội dung văn bản.
        labels (list[int]): Nhãn tương ứng.
    """
    texts, labels = [], []
    for label_idx, label_name in enumerate(["neg", "pos"]):
        subdir = os.path.join(directory, label_name)
        if not os.path.exists(subdir):
            continue
        for filename in os.listdir(subdir):
            filepath = os.path.join(subdir, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                labels.append(label_idx)
    return texts, labels


def preprocess_text(text: str) -> str:
    """Chuẩn hóa văn bản: chữ thường, loại ký tự đặc biệt."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def embed_corpus(ft_model, texts: list[str]) -> np.ndarray:
    """
    Chuyển danh sách văn bản thành ma trận embedding
    bằng ft.get_sentence_vector (trung bình word vectors).

    Returns:
        embeddings: ndarray shape (N, embedding_dim).
    """
    return np.array(
        [
            ft_model.get_sentence_vector(preprocess_text(t))
            for t in tqdm(texts, desc="Embedding")
        ],
        dtype=np.float32,
    )


class MLPClassifier(nn.Module):
    """MLP 3 tầng cho phân loại văn bản từ vector embedding."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader(
    X: np.ndarray,
    y: list[int],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> DataLoader:
    """Tạo DataLoader từ numpy array X và danh sách nhãn y."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Đánh giá mô hình, trả về (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for feat_batch, lbl_batch in loader:
            feat_batch = feat_batch.to(device)
            lbl_batch  = lbl_batch.to(device)
            out        = model(feat_batch)
            loss       = criterion(out, lbl_batch)
            total_loss += loss.item()
            correct    += (out.argmax(1) == lbl_batch).sum().item()
            total      += lbl_batch.size(0)

    return total_loss / len(loader), correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
) -> None:
    """Huấn luyện MLP với early stopping, lưu model tốt nhất."""
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        desc = f"Epoch {epoch + 1}/{num_epochs}"
        for feat_batch, lbl_batch in tqdm(train_loader, desc=desc):
            feat_batch = feat_batch.to(device)
            lbl_batch  = lbl_batch.to(device)

            optimizer.zero_grad()
            out  = model(feat_batch)
            loss = criterion(out, lbl_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (out.argmax(1) == lbl_batch).sum().item()
            total      += lbl_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc  = correct / total

        avg_val_loss, avg_val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1:>2}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {avg_val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping sau {patience} epoch không cải thiện."
                )
                break


def main():
    for zip_path in (TRAIN_ZIP, TEST_ZIP):
        if os.path.exists(zip_path):
            unzip(zip_path, DATASET_PATH)
        else:
            print(f"Không tìm thấy: {zip_path}")

    if not os.path.exists(FT_MODEL_PATH):
        print("Đang tải FastText pretrained...")
        os.system(f"wget {FT_MODEL_URL} && gunzip {FT_MODEL_PATH}.gz")
    ft = fasttext.load_model(FT_MODEL_PATH)
    print(f"Embedding dim: {ft.get_dimension()}")

    train_texts, train_labels = load_corpus_with_labels(TRAIN_DIR)
    val_texts,   val_labels   = load_corpus_with_labels(VAL_DIR)
    test_texts,  test_labels  = load_corpus_with_labels(TEST_DIR)
    print(
        f"Train: {len(train_texts)} | "
        f"Val: {len(val_texts)} | "
        f"Test: {len(test_texts)}"
    )

    print("Đang tạo embedding cho tập train...")
    X_train = embed_corpus(ft, train_texts)
    print("Đang tạo embedding cho tập val...")
    X_val   = embed_corpus(ft, val_texts)
    print("Đang tạo embedding cho tập test...")
    X_test  = embed_corpus(ft, test_texts)

    train_loader = make_loader(X_train, train_labels, shuffle=True)
    val_loader   = make_loader(X_val,   val_labels)
    test_loader  = make_loader(X_test,  test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model     = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    train(model, train_loader, val_loader, criterion, optimizer, device)

    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device)
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
