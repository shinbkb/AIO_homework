# ================================================================
# BÀI 2: DỊCH ANH → VIỆT + 3 DECODING STRATEGIES
# ================================================================
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

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = r"d:\dut_ai\AIO_code\transformer_decoder\data"
# Colab:
# from google.colab import drive; drive.mount('/content/drive')
# DATA_DIR = '/content/drive/MyDrive/transformer decoder/data'

SRC_FILE = os.path.join(DATA_DIR, "en_sents")
TGT_FILE = os.path.join(DATA_DIR, "vi_sents")

SRC_VOCAB_SIZE = 8000;  TGT_VOCAB_SIZE = 8000
D_MODEL    = 256;  NUM_HEADS  = 8;  NUM_LAYERS = 3
D_FF       = 512;  DROPOUT    = 0.1; MAX_LEN   = 64
BATCH_SIZE = 64;   NUM_EPOCHS = 10;  LR        = 1e-4
N_SAMPLES  = 50000
# ── Data ─────────────────────────────────────────────────────────
def load_pairs(src_file, tgt_file, n=None):
    with open(src_file, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    pairs = list(zip(src_lines, tgt_lines))
    if n: pairs = pairs[:n]
    print(f"Loaded {len(pairs):,} pairs"); return pairs

def tokenize(text): return text.lower().split()

def build_vocab(sentences, max_vocab=8000):
    counter = Counter()
    for s in sentences: counter.update(tokenize(s))
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for w, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[w] = len(vocab)
    return vocab

def encode(sentence, vocab, max_len=MAX_LEN):
    tokens = tokenize(sentence)[:max_len - 2]
    ids    = [vocab["<BOS>"]] + [vocab.get(t, vocab["<UNK>"]) for t in tokens] + [vocab["<EOS>"]]
    ids   += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.src = [torch.tensor(encode(s, src_vocab, max_len), dtype=torch.long) for s,_ in tqdm(pairs)]
        self.tgt = [torch.tensor(encode(t, tgt_vocab, max_len), dtype=torch.long) for _,t in tqdm(pairs)]
    def __len__(self): return len(self.src)
    def __getitem__(self, i): return self.src[i], self.tgt[i]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 2. Multi-Head Attention ───────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h, self.d_k, self.d = num_heads, d_model // num_heads, d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d)
        return self.W_o(out), attn


# ── 3. Feed-Forward Network ───────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)


# ── 4. Encoder Layer ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        a, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop1(a))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, src, src_mask=None):
        out = self.pos_enc(self.embedding(src))
        for layer in self.layers: out = layer(out, src_mask)
        return self.norm(out)


# ── 5. Decoder Layer ───────────────────────
class DecoderLayer(nn.Module):
    """
    3 sub-layer:
      1. Masked Self-Attention  → Q=K=V=tgt, dùng causal mask
      2. Cross-Attention        → Q=tgt, K=V=enc_out
      3. Feed-Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.drop1      = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)

        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3      = nn.LayerNorm(d_model)
        self.drop3      = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # Sub-layer 1: Masked Self-Attention
        a1, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt   = self.norm1(tgt + self.drop1(a1))
        # Sub-layer 2: Cross-Attention (Q từ decoder, K/V từ encoder)
        a2, _ = self.cross_attn(tgt, enc_out, enc_out, mask=src_mask)
        tgt   = self.norm2(tgt + self.drop2(a2))
        # Sub-layer 3: FFN
        tgt   = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm      = nn.LayerNorm(d_model)
    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        out = self.pos_enc(self.embedding(tgt))
        for layer in self.layers:
            out = layer(out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(out)


# ── 6. Seq2Seq Transformer ────────────────────────────────────────
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder  = TransformerEncoder(src_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder  = TransformerDecoder(tgt_vocab, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.out_proj = nn.Linear(d_model, tgt_vocab)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)


# ── 7. Mask Utilities ─────────────────────────────────────────────
def make_src_mask(src, pad_idx=0):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx=0):
    T        = tgt.size(1)
    causal   = torch.tril(torch.ones(T, T, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    return causal & pad_mask

# ── Train / Eval ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, pad_idx):
    model.train(); total = 0.0
    for src, tgt in tqdm(loader, desc="Train"):
        src, tgt        = src.to(device), tgt.to(device)
        tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_inp, make_src_mask(src, pad_idx), make_tgt_mask(tgt_inp, pad_idx))
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
            src, tgt        = src.to(device), tgt.to(device)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_inp, make_src_mask(src, pad_idx), make_tgt_mask(tgt_inp, pad_idx))
            total += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
    return total / len(loader)
# ── 3 Decoding Strategies ─────────────────────────────────────────

def greedy_decode(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN):
    """Tại mỗi bước chọn token xác suất cao nhất (argmax)."""
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)
    with torch.no_grad():
        enc = model.encoder(src, smask)
        ids = [tgt_vocab["<BOS>"]]
        for _ in range(max_len - 1):
            t   = torch.tensor([ids], dtype=torch.long).to(device)
            dec = model.decoder(t, enc, make_tgt_mask(t), smask)
            nxt = model.out_proj(dec)[0, -1].argmax(-1).item()
            ids.append(nxt)
            if nxt == tgt_vocab["<EOS>"]: break
    return " ".join(idx2w[i] for i in ids[1:] if i not in (0, 3))


def beam_decode(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN, beam_size=4):
    """Giữ beam_size hypothesis tốt nhất tại mỗi bước."""
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)
    with torch.no_grad():
        enc   = model.encoder(src, smask)
        beams = [(0.0, [tgt_vocab["<BOS>"]])]   # (log_prob, token_ids)
        done  = []
        for _ in range(max_len - 1):
            cands = []
            for lp, seq in beams:
                if seq[-1] == tgt_vocab["<EOS>"]:
                    done.append((lp, seq)); continue
                t     = torch.tensor([seq], dtype=torch.long).to(device)
                dec   = model.decoder(t, enc, make_tgt_mask(t), smask)
                log_p = F.log_softmax(model.out_proj(dec)[0, -1], dim=-1)
                for lp2, idx in zip(*log_p.topk(beam_size)):
                    cands.append((lp + lp2.item(), seq + [idx.item()]))
            if not cands: break
            beams = sorted(cands, key=lambda x: x[0], reverse=True)[:beam_size]
        best = sorted(done + beams, key=lambda x: x[0], reverse=True)[0][1]
    return " ".join(idx2w[i] for i in best[1:] if i not in (0, 3))


def sampling_decode(sentence, model, src_vocab, tgt_vocab,
                    max_len=MAX_LEN, temperature=1.0, top_k=50):
    """Lấy mẫu ngẫu nhiên từ phân phối xác suất."""
    model.eval()
    idx2w = {v: k for k, v in tgt_vocab.items()}
    src   = torch.tensor([encode(sentence, src_vocab, max_len)], dtype=torch.long).to(device)
    smask = make_src_mask(src)
    with torch.no_grad():
        enc = model.encoder(src, smask)
        ids = [tgt_vocab["<BOS>"]]
        for _ in range(max_len - 1):
            t      = torch.tensor([ids], dtype=torch.long).to(device)
            dec    = model.decoder(t, enc, make_tgt_mask(t), smask)
            logits = model.out_proj(dec)[0, -1] / temperature
            if top_k > 0:
                thresh = logits.topk(top_k).values[-1]
                logits = logits.masked_fill(logits < thresh, -float("inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            ids.append(nxt)
            if nxt == tgt_vocab["<EOS>"]: break
    return " ".join(idx2w[i] for i in ids[1:] if i not in (0, 3))
# ── Load data & Train ─────────────────────────────────────────────
pairs = load_pairs(SRC_FILE, TGT_FILE, n=N_SAMPLES)
split = int(0.9 * len(pairs))
train_pairs, val_pairs = pairs[:split], pairs[split:]

src_vocab = build_vocab([s for s,_ in train_pairs], SRC_VOCAB_SIZE)
tgt_vocab = build_vocab([t for _,t in train_pairs], TGT_VOCAB_SIZE)
SRC_VOCAB_SIZE = len(src_vocab); TGT_VOCAB_SIZE = len(tgt_vocab); PAD_IDX = 0

train_loader = DataLoader(TranslationDataset(train_pairs, src_vocab, tgt_vocab), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TranslationDataset(val_pairs,   src_vocab, tgt_vocab), BATCH_SIZE)

model = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL,
                           NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN, DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_losses, val_losses = [], []
for epoch in range(1, NUM_EPOCHS + 1):
    tl = train_epoch(model, train_loader, optimizer, criterion, PAD_IDX)
    vl = eval_epoch(model, val_loader,   criterion, PAD_IDX)
    scheduler.step(vl)
    train_losses.append(tl); val_losses.append(vl)
    print(f"Epoch {epoch:2d}: Train={tl:.4f} | Val={vl:.4f}")
# ── Plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_losses, "b-o", label="Train Loss")
plt.plot(val_losses,   "r-o", label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Transformer En→Vi"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()
# ── So sánh 3 Decoding Strategies ────────────────────────────────
tests = [
    "I love you.",
    "Tom went to school.",
    "Please close the door.",
    "She is very beautiful.",
]
for s in tests:
    print(f"\nEN          : {s}")
    print(f"Greedy      : {greedy_decode(s,   model, src_vocab, tgt_vocab)}")
    print(f"Beam (k=4)  : {beam_decode(s,     model, src_vocab, tgt_vocab, beam_size=4)}")
    print(f"Sampling    : {sampling_decode(s, model, src_vocab, tgt_vocab, temperature=0.9, top_k=50)}")
