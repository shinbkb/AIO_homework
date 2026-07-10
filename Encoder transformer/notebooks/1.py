
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

MAX_LEN    = 64
VOCAB_SIZE = 10000
D_MODEL    = 128
NUM_HEADS  = 8
NUM_LAYERS = 3
D_FF       = 256
DROPOUT    = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 5
LR         = 1e-3
DATA_PATH  = r"d:\dut_ai\AIO_code\Encoder transformer\data\sentiment_data.csv"
LABEL_NAMES = {0: "Negative", 1: "Positive", 2: "Neutral"}


def load_data(path, n_samples=50000):
    df = pd.read_csv(path)
    print(f"Toàn bộ dataset: {df.shape}")
    print(f"Phân phối nhãn:\n{df['Sentiment'].value_counts()}\n")
    return df.sample(n_samples, random_state=42).reset_index(drop=True)

def simple_tokenize(text):
    return str(text).lower().split()

def build_vocab(texts, max_vocab=10000):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = simple_tokenize(text)
    ids = [vocab["<CLS>"]]
    for token in tokens[:max_len - 1]:
        ids.append(vocab.get(token, vocab["<UNK>"]))
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.encodings = [encode_text(t, vocab, max_len) for t in tqdm(texts, desc="Encoding")]
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        self.q_linear   = nn.Linear(d_model, d_model)
        self.k_linear   = nn.Linear(d_model, d_model)
        self.v_linear   = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # Chiếu & tách head
        q = self.q_linear(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        # Ghép head & chiếu đầu ra
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_linear(out), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha      = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn      = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.drop1    = nn.Dropout(dropout)
        self.drop2    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, mask=mask)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        out = self.pos_enc(self.embedding(x))
        for layer in self.layers:
            out = layer(out, mask=mask)
        return self.norm(out)


class SentimentClassifier(nn.Module):
    """
    Dùng vector tại vị trí [CLS] (index 0) làm đại diện toàn câu,
    rồi đưa qua lớp Linear để phân loại.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff, num_classes, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.encoder    = TransformerEncoder(vocab_size, d_model, num_layers,
                                             num_heads, d_ff, max_len, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, input_ids):
        mask       = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        enc_out    = self.encoder(input_ids, mask)                # [B,L,D]
        cls_vector = enc_out[:, 0, :]                             # [B,D]
        return self.classifier(cls_vector)                        # [B,C]


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="  Train")
    for ids, labels in pbar:
        ids, labels = ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(ids)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct    += (logits.argmax(-1) == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits      = model(ids)
            total_loss += criterion(logits, labels).item()
            correct    += (logits.argmax(-1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, train_losses, "b-o", label="Train Loss")
    axes[0].plot(ep, val_losses,   "r-o", label="Val Loss")
    axes[0].set_title("Loss theo Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ep, train_accs, "b-o", label="Train Accuracy")
    axes[1].plot(ep, val_accs,   "r-o", label="Val Accuracy")
    axes[1].set_title("Accuracy theo Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)
    plt.suptitle("Transformer Encoder — Sentiment Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Đã lưu biểu đồ: training_curves.png")


def predict(text, model, vocab):
    model.eval()
    ids = torch.tensor([encode_text(text, vocab)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
        probs  = F.softmax(logits, dim=-1)[0].tolist()
        pred   = logits.argmax(-1).item()
    return LABEL_NAMES[pred], probs


if __name__ == "__main__":
    df = load_data(DATA_PATH, n_samples=50000)
    num_classes = df["Sentiment"].nunique()
    vocab = build_vocab(df["Comment"].tolist(), max_vocab=VOCAB_SIZE)
    VOCAB_SIZE = len(vocab)
    print(f"Vocab size: {VOCAB_SIZE} | Num classes: {num_classes}\n")

    # --- Split ---
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["Comment"].tolist(), df["Sentiment"].tolist(),
        test_size=0.2, random_state=42, stratify=df["Sentiment"]
    )

    train_ds = SentimentDataset(train_texts, train_labels, vocab)
    val_ds   = SentimentDataset(val_texts,   val_labels,   vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    model = SentimentClassifier(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, num_classes=num_classes, dropout=DROPOUT
    ).to(device)
    print(f"Tổng tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = eval_epoch(model,  val_loader,   criterion)
        scheduler.step()
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_accs.append(t_acc);    val_accs.append(v_acc)
        print(f"  Train → Loss: {t_loss:.4f}  Acc: {t_acc:.4f}")
        print(f"  Val   → Loss: {v_loss:.4f}  Acc: {v_acc:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs)

    print("\n--- Inference thử nghiệm ---")
    tests = [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, complete waste of money.",
        "It works fine, nothing special.",
    ]
    for sent in tests:
        label, probs = predict(sent, model, vocab)
        print(f"Text : {sent}")
        print(f"Label: {label} | Probs: {[f'{p:.2%}' for p in probs]}\n")
