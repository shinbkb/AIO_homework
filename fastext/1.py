import os
import re
import zipfile
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Đường dẫn dữ liệu
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

# Siêu tham số
MAX_SEQ_LEN          = 100    # Số từ tối đa mỗi tài liệu
MAX_SUBWORD_PER_WORD = 20     # Số subword tối đa mỗi từ
N_GRAM_MIN           = 3
N_GRAM_MAX           = 5
MIN_WORD_FREQ        = 5      # Tần suất tối thiểu để đưa vào vocab
EMBEDDING_DIM        = 100
NUM_CLASSES          = 2      # 0: tiêu cực, 1: tích cực
BATCH_SIZE           = 64
LEARNING_RATE        = 0.001
NUM_EPOCHS           = 50
PATIENCE             = 5      # Early stopping
MIN_DELTA            = 0.001  # Cải thiện tối thiểu để tiếp tục huấn luyện


def unzip(zip_path: str, extract_to: str) -> None:
    """Giải nén file zip vào thư mục chỉ định."""
    print(f"Đang giải nén {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Hoàn thành giải nén vào {extract_to}")


def load_corpus_with_labels(base_directory: str):
    """
    Đọc toàn bộ văn bản và nhãn từ cấu trúc thư mục:
      base_directory/
        pos/  -> nhãn 1 (tích cực)
        neg/  -> nhãn 0 (tiêu cực)

    Returns:
        corpus (list[str]): Danh sách nội dung văn bản.
        labels (list[int]): Danh sách nhãn tương ứng.
    """
    corpus, labels = [], []
    for root, _, files in os.walk(base_directory):
        parts = root.split(os.sep)
        if "pos" in parts:
            label = 1
        elif "neg" in parts:
            label = 0
        else:
            continue

        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    corpus.append(f.read())
                    labels.append(label)

    return corpus, labels


def preprocess_text(text: str) -> list[str]:
    """Chuẩn hóa văn bản: chữ thường, loại ký tự đặc biệt, tách từ."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def build_vocab(tokenized_corpus: list[list[str]], min_freq: int = 5):
    """
    Xây dựng từ điển từ tập dữ liệu đã token hóa.

    Args:
        tokenized_corpus: Danh sách các tài liệu đã token hóa.
        min_freq: Ngưỡng tần suất tối thiểu.

    Returns:
        word_to_idx (dict): Ánh xạ từ -> index (0 = <unk>).
        vocab (list): Danh sách từ trong từ điển.
        word_counts (Counter): Tần suất của từng từ.
    """
    word_counts = Counter()
    for doc_tokens in tokenized_corpus:
        word_counts.update(doc_tokens)

    vocab = [w for w, c in word_counts.items() if c >= min_freq]
    word_to_idx = {w: idx + 1 for idx, w in enumerate(vocab)}
    word_to_idx["<unk>"] = 0
    return word_to_idx, vocab, word_counts


def get_ngrams(word: str, n: int) -> list[str]:
    """Tạo danh sách n-gram ký tự có ranh giới '<' và '>' cho một từ."""
    bounded = f"<{word}>"
    return [bounded[i:i + n] for i in range(len(bounded) - n + 1)]


def build_subword_vocab(
    word_to_idx: dict,
    n_gram_min: int = 3,
    n_gram_max: int = 5,
) -> dict:
    """
    Xây dựng từ điển subword (n-gram ký tự) từ word vocab.

    Returns:
        subword_to_idx (dict): Ánh xạ subword -> index (0 = <unk_sub>).
    """
    subword_to_idx = {}
    idx = 1  # 0 dành cho PAD/UNK subword
    for word in word_to_idx:
        if word == "<unk>":
            continue
        for n in range(n_gram_min, n_gram_max + 1):
            for ng in get_ngrams(word, n):
                if ng not in subword_to_idx:
                    subword_to_idx[ng] = idx
                    idx += 1
    subword_to_idx["<unk_sub>"] = 0
    return subword_to_idx


def tokens_to_indices(
    tokenized_corpus: list[list[str]],
    word_to_idx: dict,
) -> torch.Tensor:
    """
    Chuyển token thành index, padding/truncation về MAX_SEQ_LEN.

    Returns:
        Tensor shape (num_docs, MAX_SEQ_LEN).
    """
    unk_idx = word_to_idx["<unk>"]
    indexed = []
    for doc_tokens in tokenized_corpus:
        indices = [word_to_idx.get(t, unk_idx) for t in doc_tokens]
        if len(indices) < MAX_SEQ_LEN:
            indices += [0] * (MAX_SEQ_LEN - len(indices))
        else:
            indices = indices[:MAX_SEQ_LEN]
        indexed.append(indices)
    return torch.tensor(indexed, dtype=torch.long)


def subword_to_indices(
    tokenized_corpus: list[list[str]],
    subword_to_idx: dict,
) -> torch.Tensor:
    """
    Chuyển token thành index subword, padding/truncation về
    (MAX_SEQ_LEN, MAX_SUBWORD_PER_WORD).

    Returns:
        Tensor shape (num_docs, MAX_SEQ_LEN, MAX_SUBWORD_PER_WORD).
    """
    unk_sub  = subword_to_idx["<unk_sub>"]
    indexed  = []
    pad_word = [0] * MAX_SUBWORD_PER_WORD

    for doc_tokens in tokenized_corpus:
        doc_subword = []
        for token in doc_tokens:
            word_subs = []
            for n in range(N_GRAM_MIN, N_GRAM_MAX + 1):
                word_subs.extend(
                    subword_to_idx.get(ng, unk_sub)
                    for ng in get_ngrams(token, n)
                )
            if len(word_subs) < MAX_SUBWORD_PER_WORD:
                word_subs += [0] * (MAX_SUBWORD_PER_WORD - len(word_subs))
            else:
                word_subs = word_subs[:MAX_SUBWORD_PER_WORD]
            doc_subword.append(word_subs)

        if len(doc_subword) < MAX_SEQ_LEN:
            pad_count = MAX_SEQ_LEN - len(doc_subword)
            doc_subword += [pad_word[:] for _ in range(pad_count)]
        else:
            doc_subword = doc_subword[:MAX_SEQ_LEN]

        indexed.append(doc_subword)

    return torch.tensor(indexed, dtype=torch.long)


def build_dataloaders(
    train_tokens, val_tokens, test_tokens,
    train_labels, val_labels, test_labels,
    word_to_idx, subword_to_idx,
):
    """Tạo TensorDataset và DataLoader cho train/val/test."""
    train_words = tokens_to_indices(train_tokens, word_to_idx)
    val_words   = tokens_to_indices(val_tokens,   word_to_idx)
    test_words  = tokens_to_indices(test_tokens,  word_to_idx)

    train_subs = subword_to_indices(train_tokens, subword_to_idx)
    val_subs   = subword_to_indices(val_tokens,   subword_to_idx)
    test_subs  = subword_to_indices(test_tokens,  subword_to_idx)

    train_lbl = torch.tensor(train_labels, dtype=torch.long)
    val_lbl   = torch.tensor(val_labels,   dtype=torch.long)
    test_lbl  = torch.tensor(test_labels,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_words, train_subs, train_lbl),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_words, val_subs, val_lbl),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        TensorDataset(test_words, test_subs, test_lbl),
        batch_size=BATCH_SIZE,
    )

    print(f"Train words shape   : {train_words.shape}")
    print(f"Train subwords shape: {train_subs.shape}")
    print(f"Train labels shape  : {train_lbl.shape}")
    print(f"Total train batches : {len(train_loader)}")

    return train_loader, val_loader, test_loader


class FastText(nn.Module):
    """
    Mô hình FastText cho phân loại văn bản.

    Kết hợp word embeddings và subword (character n-gram) embeddings,
    lấy trung bình cộng có mặt nạ (masked average) để tạo biểu diễn
    toàn tài liệu, sau đó dự đoán nhãn qua tầng Linear → Softmax.
    """

    def __init__(
        self,
        vocab_size: int,
        subword_vocab_size: int,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.subword_embeddings = nn.Embedding(
            subword_vocab_size + 1, embedding_dim, padding_idx=0
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        word_ids: torch.Tensor,
        subword_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            word_ids   : (batch, max_seq_len)
            subword_ids: (batch, max_seq_len, max_subword_per_word)

        Returns:
            logits: (batch, num_classes)
        """
        # Word embeddings: (B, L, D)
        word_embeds = self.word_embeddings(word_ids)

        # Subword embeddings
        B, L, S = subword_ids.shape
        subword_embeds = self.subword_embeddings(
            subword_ids.view(-1)
        ).view(B, L, S, self.embedding_dim)

        # Masked average qua chiều subword: (B, L, D)
        sub_mask = (subword_ids != 0).float().unsqueeze(-1)
        avg_sub  = (subword_embeds * sub_mask).sum(dim=2)
        avg_sub  = avg_sub / sub_mask.sum(dim=2).clamp(min=1)

        # Kết hợp word + subword: (B, L, D)
        combined = word_embeds + avg_sub

        # Masked average qua chiều từ → biểu diễn tài liệu: (B, D)
        word_mask = (word_ids != 0).float().unsqueeze(-1)
        doc_embed = (combined * word_mask).sum(dim=1)
        doc_embed = doc_embed / word_mask.sum(dim=1).clamp(min=1)

        # Dự đoán: Linear → Softmax (CE loss dùng logits)
        return self.fc(doc_embed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Đánh giá mô hình trên một DataLoader, trả về (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for word_batch, sub_batch, lbl_batch in loader:
            word_batch = word_batch.to(device)
            sub_batch  = sub_batch.to(device)
            lbl_batch  = lbl_batch.to(device)
            outputs    = model(word_batch, sub_batch)
            total_loss += criterion(outputs, lbl_batch).item()
            correct    += (outputs.argmax(dim=1) == lbl_batch).sum().item()
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
    min_delta: float = MIN_DELTA,
) -> None:
    """Huấn luyện mô hình với early stopping."""
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        desc = f"Epoch {epoch + 1}/{num_epochs}"
        for word_batch, sub_batch, lbl_batch in tqdm(train_loader, desc=desc):
            word_batch = word_batch.to(device)
            sub_batch  = sub_batch.to(device)
            lbl_batch  = lbl_batch.to(device)

            optimizer.zero_grad()
            outputs = model(word_batch, sub_batch)
            loss    = criterion(outputs, lbl_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(dim=1) == lbl_batch).sum().item()
            total      += lbl_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:>2}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss     = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping sau {patience} epoch không cải thiện.")
                break


def main():
    for zip_path in (TRAIN_ZIP, TEST_ZIP):
        if os.path.exists(zip_path):
            unzip(zip_path, DATASET_PATH)
        else:
            print(f"Không tìm thấy: {zip_path}")

    train_corpus, train_labels = load_corpus_with_labels(TRAIN_DIR)
    val_corpus,   val_labels   = load_corpus_with_labels(VAL_DIR)
    test_corpus,  test_labels  = load_corpus_with_labels(TEST_DIR)

    print(
        f"Train: {len(train_corpus)} docs | "
        f"Val: {len(val_corpus)} docs | "
        f"Test: {len(test_corpus)} docs"
    )

    train_tokens = [preprocess_text(doc) for doc in train_corpus]
    val_tokens   = [preprocess_text(doc) for doc in val_corpus]
    test_tokens  = [preprocess_text(doc) for doc in test_corpus]

    word_to_idx, vocab, word_counts = build_vocab(
        train_tokens, min_freq=MIN_WORD_FREQ
    )
    subword_to_idx = build_subword_vocab(word_to_idx, N_GRAM_MIN, N_GRAM_MAX)

    print(f"Vocab size        : {len(vocab)}")
    print(f"Subword vocab size: {len(subword_to_idx)}")
    print(f"Top 10 từ phổ biến: {word_counts.most_common(10)}")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_tokens, val_tokens, test_tokens,
        train_labels, val_labels, test_labels,
        word_to_idx, subword_to_idx,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    model = FastText(
        vocab_size=len(word_to_idx),
        subword_vocab_size=len(subword_to_idx),
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, val_loader, criterion, optimizer, device)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
