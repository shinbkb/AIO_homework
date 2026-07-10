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
DATA_DIR = r"d:\dut_ai\AIO_code\Dence Representation\data\data_train\train"
EMBED_DIM = 100
EPOCHS = 5
LR = 0.001
BATCH_SIZE = 1024
WINDOW = 2
MIN_FREQ = 3
SUBSET_SIZE = 1000000 # Giới hạn mẫu để tránh lỗi RAM

def load_corpus(data_dir):
    texts = []
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.txt'):
                all_files.append(os.path.join(root, fname))
    
    print("Loading corpus...")
    for fpath in tqdm(all_files, desc="Reading files"):
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                texts.append(text)
    return texts

def preprocess(texts, min_freq=3):
    print("Preprocessing texts...")
    all_tokens = []
    tokenized_docs = []
    for text in tqdm(texts, desc="Tokenizing"):
        text = text.lower()
        # Giữ lại chữ cái tiếng Việt và dấu _ cho từ ghép
        tokens = re.findall(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ_]+', text)
        tokenized_docs.append(tokens)
        all_tokens.extend(tokens)
    
    freq = Counter(all_tokens)
    vocab = ['<UNK>'] + [w for w, c in freq.items() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    return tokenized_docs, vocab, word2idx, idx2word

def generate_skipgram_data(tokenized_docs, word2idx, window=2):
    print("Generating Skip-gram pairs (Center -> Context)...")
    data = []
    for tokens in tqdm(tokenized_docs, desc="Extracting pairs"):
        ids = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
        for i in range(len(ids)):
            center_id = ids[i]
            start = max(0, i - window)
            end = min(len(ids), i + window + 1)
            for j in range(start, end):
                if i == j: continue
                context_id = ids[j]
                data.append((center_id, context_id))
    return data

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear    = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, center_id):
        # center_id shape: (batch_size)
        embed = self.embedding(center_id) # (batch, embed_dim)
        out   = self.linear(embed)       # (batch, vocab_size)
        return out

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load & Preprocess
    corpus = load_corpus(DATA_DIR)
    tokenized_docs, vocab, word2idx, idx2word = preprocess(corpus, min_freq=MIN_FREQ)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # 2. Sinh dữ liệu Skip-gram
    data = generate_skipgram_data(tokenized_docs, word2idx, window=WINDOW)
    print(f"Total samples: {len(data)}")

    # 3. Chia tập Train/Val và lấy Subset
    subset_data = data[:SUBSET_SIZE]
    train_data, val_data = train_test_split(subset_data, test_size=0.2, random_state=42)

    def create_loader(data_list, batch_size):
        X = torch.tensor([center for center, ctx in data_list], dtype=torch.long)
        y = torch.tensor([ctx for center, ctx in data_list], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    train_loader = create_loader(train_data, BATCH_SIZE)
    val_loader = create_loader(val_data, BATCH_SIZE)

    # 4. Khởi tạo Mô hình
    model = SkipGram(vocab_size, EMBED_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            _, preds = torch.max(out, 1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total:.4f}"})

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, 1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/train_total:.4f}, Train Acc: {train_correct/train_total:.4f} | "
              f"Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_correct/val_total:.4f}\n")

    # 6. Lưu kết quả
    print("Saving results...")
    torch.save(model.state_dict(), "skipgram_model.pth")
    torch.save(model.embedding.weight.data.cpu(), "skipgram_embeddings.pt")
    with open("skipgram_vocab.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    print("Xong! Kết quả đã được lưu.")

if __name__ == "__main__":
    train()
