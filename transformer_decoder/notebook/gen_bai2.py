import nbformat

nb = nbformat.v4.new_notebook()
cells = []

# ── MARKDOWN: tiêu đề ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
# Bài 2: Dịch Ngôn Ngữ Anh → Việt với Transformer
**Yêu cầu:** Sử dụng kiến trúc Transformer để dịch tiếng Anh sang tiếng Việt.
So sánh **3 Decoding Strategies**: Greedy Search, Beam Search, Sampling.
"""))

# ── CELL 1: imports + config ──────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Đường dẫn data ───────────────────────────────────────────────────────────
# Local:
DATA_DIR = r"d:\\dut_ai\\AIO_code\\transformer_decoder\\data"
# Colab (uncomment nếu chạy trên Colab):
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/transformer decoder/data'

SRC_FILE = os.path.join(DATA_DIR, "en_sents")
TGT_FILE = os.path.join(DATA_DIR, "vi_sents")
print("en_sents:", os.path.exists(SRC_FILE))
print("vi_sents:", os.path.exists(TGT_FILE))

# ── Hyperparameters ───────────────────────────────────────────────────────────
SRC_VOCAB_SIZE = 8000
TGT_VOCAB_SIZE = 8000
D_MODEL    = 256
NUM_HEADS  = 8
NUM_LAYERS = 3
D_FF       = 512
DROPOUT    = 0.1
MAX_LEN    = 64
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR         = 1e-4
N_SAMPLES  = 50000
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 1. Data Loading & Vocabulary"))

# ── CELL 2: data utils ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
def load_pairs(src_file, tgt_file, n=None):
    with open(src_file, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    pairs = list(zip(src_lines, tgt_lines))
    if n: pairs = pairs[:n]
    print(f"Loaded {len(pairs):,} pairs")
    return pairs

def tokenize(text):
    return text.lower().split()

def build_vocab(sentences, max_vocab=8000):
    counter = Counter()
    for s in sentences: counter.update(tokenize(s))
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for w, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[w] = len(vocab)
    return vocab

def encode(sentence, vocab, max_len=MAX_LEN):
    tokens = tokenize(sentence)[:max_len - 2]
    ids = ([vocab["<BOS>"]]
           + [vocab.get(t, vocab["<UNK>"]) for t in tokens]
           + [vocab["<EOS>"]])
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.src = [torch.tensor(encode(s, src_vocab, max_len), dtype=torch.long)
                    for s, _ in tqdm(pairs, desc="Encoding src")]
        self.tgt = [torch.tensor(encode(t, tgt_vocab, max_len), dtype=torch.long)
                    for _, t in tqdm(pairs, desc="Encoding tgt")]
    def __len__(self): return len(self.src)
    def __getitem__(self, idx): return self.src[idx], self.tgt[idx]
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 2. Model Architecture (từ Bài 1)"))

# ── CELL 3: tất cả model modules ──────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads; self.d_k = d_model // num_heads; self.d = d_model
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.W_q(q).view(B,-1,self.h,self.d_k).transpose(1,2)
        k = self.W_k(k).view(B,-1,self.h,self.d_k).transpose(1,2)
        v = self.W_v(v).view(B,-1,self.h,self.d_k).transpose(1,2)
        sc = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None: sc = sc.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(sc, dim=-1))
        out  = torch.matmul(attn, v).transpose(1,2).contiguous().view(B,-1,self.d)
        return self.W_o(out), attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff,d_model))
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout);   self.drop2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, mask=mask)
        x = self.norm1(x + self.drop1(a))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos    = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(d_model)
    def forward(self, src, mask=None):
        out = self.pos(self.emb(src))
        for layer in self.layers: out = layer(out, mask)
        return self.norm(out)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model); self.drop1 = nn.Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model); self.drop2 = nn.Dropout(dropout)
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model); self.drop3 = nn.Dropout(dropout)
    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        a1, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt   = self.norm1(tgt + self.drop1(a1))
        a2, _ = self.cross_attn(tgt, enc_out, enc_out, mask=src_mask)
        tgt   = self.norm2(tgt + self.drop2(a2))
        tgt   = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.emb    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos    = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(d_model)
    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        out = self.pos(self.emb(tgt))
        for layer in self.layers: out = layer(out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(out)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder  = TransformerEncoder(src_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder  = TransformerDecoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.out_proj = nn.Linear(d_model, tgt_vocab)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, src_mask)
        return self.out_proj(dec)

def make_src_mask(src, pad=0):
    return (src != pad).unsqueeze(1).unsqueeze(2)           # [B,1,1,S]

def make_tgt_mask(tgt, pad=0):
    T        = tgt.size(1)
    causal   = torch.tril(torch.ones(T, T, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    pad_mask = (tgt != pad).unsqueeze(1).unsqueeze(2)
    return causal & pad_mask                                # [B,1,T,T]

print("Model classes defined!")
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 3. Training"))

# ── CELL 4: train/eval ────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
def train_epoch(model, loader, optimizer, criterion, pad_idx):
    model.train(); total = 0.0
    for src, tgt in tqdm(loader, desc="Train"):
        src, tgt = src.to(device), tgt.to(device)
        tgt_inp  = tgt[:, :-1]; tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx)
        optimizer.zero_grad()
        logits = model(src, tgt_inp, src_mask, tgt_mask)
        loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, criterion, pad_idx):
    model.eval(); total = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_inp  = tgt[:, :-1]; tgt_out = tgt[:, 1:]
            logits   = model(src, tgt_inp,
                             make_src_mask(src, pad_idx),
                             make_tgt_mask(tgt_inp, pad_idx))
            total   += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
    return total / len(loader)
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 4. Decoding Strategies
So sánh 3 chiến lược giải mã:

| Strategy | Cách chọn token | Đặc điểm |
|---|---|---|
| **Greedy Search** | `argmax` tại mỗi bước | Nhanh, nhưng có thể bỏ lỡ kết quả tốt hơn |
| **Beam Search** | Giữ `k` hypothesis tốt nhất | Cân bằng giữa tốc độ và chất lượng |
| **Sampling** | Lấy mẫu ngẫu nhiên từ phân phối | Đa dạng, sáng tạo hơn |
"""))

# ── CELL 5: 3 decoding strategies ────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ─── 1. GREEDY SEARCH ────────────────────────────────────────────────────────
def greedy_decode(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN):
    \"\"\"Tại mỗi bước chọn token có xác suất cao nhất (argmax).\"\"\"
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)
    with torch.no_grad():
        enc     = model.encoder(src, smask)
        tgt_ids = [tgt_vocab["<BOS>"]]
        for _ in range(max_len - 1):
            t     = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            tmask = make_tgt_mask(t)
            dec   = model.decoder(t, enc, tmask, smask)
            nxt   = model.out_proj(dec)[0, -1].argmax(-1).item()
            tgt_ids.append(nxt)
            if nxt == tgt_vocab["<EOS>"]: break
    tokens = [idx2w.get(i, "<UNK>") for i in tgt_ids[1:]
              if i not in (tgt_vocab["<PAD>"], tgt_vocab["<EOS>"])]
    return " ".join(tokens)


# ─── 2. BEAM SEARCH ──────────────────────────────────────────────────────────
def beam_decode(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN, beam_size=4):
    \"\"\"
    Giữ beam_size hypothesis tốt nhất tại mỗi bước.
    Mỗi hypothesis: (log_prob_tổng, [danh_sách_token_ids])
    \"\"\"
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)

    with torch.no_grad():
        enc = model.encoder(src, smask)

        # Khởi tạo: 1 hypothesis ban đầu với token <BOS>
        beams = [(0.0, [tgt_vocab["<BOS>"]])]   # (log_prob, token_ids)
        completed = []

        for _ in range(max_len - 1):
            candidates = []
            for log_prob, seq in beams:
                if seq[-1] == tgt_vocab["<EOS>"]:
                    completed.append((log_prob, seq))
                    continue
                t     = torch.tensor([seq], dtype=torch.long).to(device)
                tmask = make_tgt_mask(t)
                dec   = model.decoder(t, enc, tmask, smask)
                # Log-prob của tất cả token tại bước cuối
                log_probs = F.log_softmax(model.out_proj(dec)[0, -1], dim=-1)
                # Lấy top beam_size token
                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, idx in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((log_prob + lp, seq + [idx]))

            if not candidates: break
            # Chọn beam_size hypothesis tốt nhất (score cao nhất)
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        # Lấy hypothesis tốt nhất
        completed += beams
        best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]

    tokens = [idx2w.get(i, "<UNK>") for i in best_seq[1:]
              if i not in (tgt_vocab["<PAD>"], tgt_vocab["<EOS>"])]
    return " ".join(tokens)


# ─── 3. SAMPLING ─────────────────────────────────────────────────────────────
def sampling_decode(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN,
                    temperature=1.0, top_k=50):
    \"\"\"
    Lấy mẫu ngẫu nhiên từ phân phối xác suất.
    - temperature: < 1 → nhọn hơn (ít đa dạng); > 1 → phẳng hơn (đa dạng hơn)
    - top_k: chỉ lấy mẫu trong top_k token có xác suất cao nhất
    \"\"\"
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)

    with torch.no_grad():
        enc     = model.encoder(src, smask)
        tgt_ids = [tgt_vocab["<BOS>"]]
        for _ in range(max_len - 1):
            t     = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            tmask = make_tgt_mask(t)
            dec   = model.decoder(t, enc, tmask, smask)
            logits = model.out_proj(dec)[0, -1] / temperature   # scale by temperature

            # Top-k filtering: đặt các token ngoài top_k về -inf
            if top_k > 0:
                threshold = logits.topk(top_k).values[-1]
                logits = logits.masked_fill(logits < threshold, -float("inf"))

            probs = F.softmax(logits, dim=-1)
            nxt   = torch.multinomial(probs, num_samples=1).item()   # lấy mẫu ngẫu nhiên
            tgt_ids.append(nxt)
            if nxt == tgt_vocab["<EOS>"]: break

    tokens = [idx2w.get(i, "<UNK>") for i in tgt_ids[1:]
              if i not in (tgt_vocab["<PAD>"], tgt_vocab["<EOS>"])]
    return " ".join(tokens)

print("3 Decoding strategies defined!")
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 5. Load Data & Train"))

# ── CELL 6: load data + train ─────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
pairs = load_pairs(SRC_FILE, TGT_FILE, n=N_SAMPLES)
split = int(0.9 * len(pairs))
train_pairs, val_pairs = pairs[:split], pairs[split:]

src_vocab = build_vocab([s for s,_ in train_pairs], SRC_VOCAB_SIZE)
tgt_vocab = build_vocab([t for _,t in train_pairs], TGT_VOCAB_SIZE)
SRC_VOCAB_SIZE = len(src_vocab); TGT_VOCAB_SIZE = len(tgt_vocab)
PAD_IDX = 0
print(f"src_vocab={SRC_VOCAB_SIZE} | tgt_vocab={TGT_VOCAB_SIZE}")

train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL,
                           NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN, DROPOUT).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.98), eps=1e-9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_losses, val_losses = [], []
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\\nEpoch {epoch}/{NUM_EPOCHS}")
    tl = train_epoch(model, train_loader, optimizer, criterion, PAD_IDX)
    vl = eval_epoch(model, val_loader,   criterion, PAD_IDX)
    scheduler.step(vl)
    train_losses.append(tl); val_losses.append(vl)
    print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f}")
"""))

# ── CELL 7: plot ──────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
plt.figure(figsize=(8,4))
plt.plot(train_losses, "b-o", label="Train Loss")
plt.plot(val_losses,   "r-o", label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Transformer — En→Vi Translation")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 6. So sánh 3 Decoding Strategies"))

# ── CELL 8: compare strategies ────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
test_sentences = [
    "I love you.",
    "How are you?",
    "Tom went to school.",
    "She is very beautiful.",
    "Please close the door.",
    "We have to save Tom.",
    "Tom doesn't like eating vegetables.",
]

print("=" * 70)
print(f"{'Sentence':<35} {'Strategy':<12} {'Translation'}")
print("=" * 70)

for sent in test_sentences:
    g = greedy_decode(sent,   model, src_vocab, tgt_vocab)
    b = beam_decode(sent,     model, src_vocab, tgt_vocab, beam_size=4)
    s = sampling_decode(sent, model, src_vocab, tgt_vocab, temperature=0.9, top_k=50)
    print(f"\\nEN : {sent}")
    print(f"  Greedy   : {g}")
    print(f"  Beam(k=4): {b}")
    print(f"  Sampling : {s}")
"""))

# ── CELL 9: detailed comparison table ────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# So sánh Beam Search với các beam_size khác nhau
sentence = "Tom didn't understand exactly what Mary wanted him to do."
print(f"Câu gốc: {sentence}\\n")

print(f"{'Strategy':<25} {'Kết quả dịch'}")
print("-" * 80)
print(f"{'Greedy':<25} {greedy_decode(sentence, model, src_vocab, tgt_vocab)}")
for k in [2, 4, 8]:
    r = beam_decode(sentence, model, src_vocab, tgt_vocab, beam_size=k)
    print(f"{'Beam (k='+str(k)+')':<25} {r}")
for t in [0.5, 1.0, 1.5]:
    r = sampling_decode(sentence, model, src_vocab, tgt_vocab, temperature=t)
    print(f"{'Sampling (T='+str(t)+')':<25} {r}")
"""))

nb.cells = cells

import os
out_path = r"d:\dut_ai\AIO_code\transformer_decoder\notebook\2.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Done! {out_path}")
