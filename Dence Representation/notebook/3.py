import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH (CONFIGURATION) ---
TRAIN_DIR = r"d:\dut_ai\AIO_code\Dence Representation\data\data_train\train"
TEST_DIR  = r"d:\dut_ai\AIO_code\Dence Representation\data\data_test\test"
EMBED_DIM = 100
MAX_LEN   = 50
EPOCHS    = 10
LR        = 0.001
BATCH_SIZE = 128
MIN_FREQ  = 3

def load_labeled_data(data_dir):
    texts = []
    labels = []
    # Quy ước: pos = 1, neg = 0
    for label_name, label_idx in [('pos', 1), ('neg', 0)]:
        folder_path = os.path.join(data_dir, label_name)
        if not os.path.exists(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        print(f"Loading {label_name} files from {data_dir}...")
        for fname in tqdm(files, desc=label_name):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
                    labels.append(label_idx)
    return texts, labels

def tokenize(text):
    text = text.lower()
    # Giữ lại chữ cái tiếng Việt và dấu _ cho từ ghép
    return re.findall(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ_]+', text)

def build_vocab(texts, min_freq=3):
    print("Building vocabulary...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing train set"):
        all_tokens.extend(tokenize(text))
    
    freq = Counter(all_tokens)
    # <PAD> dùng cho padding, <UNK> dùng cho từ lạ
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in freq.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx

def texts_to_sequences(texts, word2idx, max_len):
    sequences = []
    for text in tqdm(texts, desc="Padding/Truncating"):
        tokens = tokenize(text)
        indices = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
        # Padding hoặc Truncating
        if len(indices) < max_len:
            indices = indices + [word2idx['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        sequences.append(indices)
    return torch.tensor(sequences, dtype=torch.long)

class SentimentMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(max_len * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.embedding(x)      # (batch, max_len, embed_dim)
        x = self.flatten(x)        # (batch, max_len * embed_dim)
        return self.mlp(x)         # (batch, 1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    train_texts, train_labels = load_labeled_data(TRAIN_DIR)
    test_texts, test_labels = load_labeled_data(TEST_DIR)

    # 2. Build Vocab & Sequences
    word2idx = build_vocab(train_texts, min_freq=MIN_FREQ)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    X_train_full = texts_to_sequences(train_texts, word2idx, MAX_LEN)
    y_train_full = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    
    X_test = texts_to_sequences(test_texts, word2idx, MAX_LEN)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    # 3. Train/Val Split (80/20)
    indices = np.arange(len(X_train_full))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_train_full.numpy())
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_full[train_idx], y_train_full[train_idx]), 
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_full[val_idx], y_train_full[val_idx]), 
        batch_size=BATCH_SIZE
    )

    # 4. Model setup
    model = SentimentMLP(vocab_size, EMBED_DIM, MAX_LEN).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            preds = (out > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            pbar.set_postfix({"acc": f"{correct/total:.4f}"})

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = (out > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/total:.4f}, Train Acc: {correct/total:.4f} | Val Acc: {val_correct/val_total:.4f}\n")

    # 6. Đánh giá cuối cùng trên tập Test
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for xb, yb in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=BATCH_SIZE):
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = (out > 0.5).float()
            test_correct += (preds == yb).sum().item()
            test_total += yb.size(0)
    print(f"Final Test Accuracy: {test_correct/test_total:.4f}")

if __name__ == "__main__":
    import numpy as np
    train()
