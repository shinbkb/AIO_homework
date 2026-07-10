
# =============================================================================
# BÀI 1: TRANSFORMER DECODER NHIỀU LỚP (KHÔNG DÙNG nn.TransformerDecoderLayer)
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATA_DIR    = r"d:\dut_ai\AIO_code\transformer_decoder\data"
SRC_FILE    = DATA_DIR + r"\en_sents"
TGT_FILE    = DATA_DIR + r"\vi_sents"

SRC_VOCAB   = 8000
TGT_VOCAB   = 8000
D_MODEL     = 256
NUM_HEADS   = 8          # d_model // num_heads = 32
NUM_LAYERS  = 3
D_FF        = 512
DROPOUT     = 0.1
MAX_LEN     = 64
BATCH_SIZE  = 64
NUM_EPOCHS  = 10
LR          = 1e-4
N_SAMPLES   = 50000      # số cặp câu sử dụng


# =============================================================================
# 1. DATA LOADING & VOCABULARY
# =============================================================================

def load_pairs(src_file, tgt_file, n=None):
    with open(src_file, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]
    pairs = list(zip(src_lines, tgt_lines))
    if n:
        pairs = pairs[:n]
    print(f"Loaded {len(pairs):,} sentence pairs")
    return pairs


def tokenize(text):
    return text.lower().split()


def build_vocab(sentences, max_vocab=8000):
    """Xây dựng vocabulary từ danh sách câu."""
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))
    # Các token đặc biệt
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def encode(sentence, vocab, max_len=MAX_LEN):
    """Mã hoá câu thành list id, thêm <BOS> và <EOS>."""
    tokens = tokenize(sentence)[:max_len - 2]
    ids = [vocab["<BOS>"]] + [vocab.get(t, vocab["<UNK>"]) for t in tokens] + [vocab["<EOS>"]]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids[:max_len]


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=MAX_LEN):
        print("Encoding dataset...")
        self.src = [torch.tensor(encode(s, src_vocab, max_len), dtype=torch.long)
                    for s, _ in tqdm(pairs)]
        self.tgt = [torch.tensor(encode(t, tgt_vocab, max_len), dtype=torch.long)
                    for _, t in tqdm(pairs)]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


# =============================================================================
# 2. MODEL MODULES
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.q_proj    = nn.Linear(d_model, d_model)
        self.k_proj    = nn.Linear(d_model, d_model)
        self.v_proj    = nn.Linear(d_model, d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # Project & split heads: [B, L, d_model] -> [B, h, L, d_k]
        q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        # Merge heads: [B, h, L, d_k] -> [B, L, d_model]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.out_proj(out), attn


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


# ── ENCODER LAYER (giống bài Encoder) ────────────────────────────────────────
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
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        out = self.pos_enc(self.embedding(src))
        for layer in self.layers:
            out = layer(out, src_mask)
        return self.norm(out)   # [B, S, D]


# ── DECODER LAYER (CẦN XÂY DỰNG MỚI) ────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    Một lớp Decoder gồm 3 sub-layer:
      1. Masked Self-Attention     (nhìn vào target, có causal mask)
      2. Cross-Attention           (Q từ decoder, K/V từ encoder output)
      3. Feed-Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)

        # Sub-layer 2: Cross-Attention (Encoder-Decoder Attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)

        # Sub-layer 3: Feed-Forward
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention: decoder tự chú ý chính nó (có causal mask)
        attn1, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.drop1(attn1))

        # 2. Cross-Attention: Q=decoder output, K=V=encoder output
        attn2, _ = self.cross_attn(tgt, enc_out, enc_out, mask=src_mask)
        tgt = self.norm2(tgt + self.drop2(attn2))

        # 3. Feed-Forward
        tgt = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        out = self.pos_enc(self.embedding(tgt))   # [B, T, D]
        for layer in self.layers:
            out = layer(out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(out)   # [B, T, D]


# ── FULL SEQ2SEQ TRANSFORMER ──────────────────────────────────────────────────
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model, num_layers, num_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.encoder  = TransformerEncoder(src_vocab_size, d_model, num_layers,
                                           num_heads, d_ff, max_len, dropout)
        self.decoder  = TransformerDecoder(tgt_vocab_size, d_model, num_layers,
                                           num_heads, d_ff, max_len, dropout)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)                      # [B, S, D]
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)   # [B, T, D]
        return self.out_proj(dec_out)                              # [B, T, tgt_vocab]


# =============================================================================
# 3. MASK UTILITIES
# =============================================================================

def make_src_mask(src, pad_idx=0):
    """Mask padding tokens trong source. Shape: [B, 1, 1, S]"""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx=0):
    """
    Kết hợp padding mask + causal mask cho target.
    Shape: [B, 1, T, T]
    """
    T = tgt.size(1)
    # Causal mask: tam giác dưới [1, 1, T, T]
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).unsqueeze(0).unsqueeze(0)
    # Padding mask: [B, 1, 1, T]
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # Kết hợp: vị trí hợp lệ phải đồng thời không phải padding VÀ trong tam giác dưới
    return causal & pad_mask


# =============================================================================
# 4. TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train")
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing: decoder nhận tgt[:, :-1], nhãn là tgt[:, 1:]
        tgt_inp = tgt[:, :-1]   # [B, T-1]  — bỏ token cuối
        tgt_out = tgt[:, 1:]    # [B, T-1]  — bỏ <BOS>

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx)

        optimizer.zero_grad()
        logits = model(src, tgt_inp, src_mask, tgt_mask)   # [B, T-1, V]

        # Tính loss, bỏ qua padding
        logits_flat = logits.reshape(-1, logits.size(-1))  # [B*(T-1), V]
        tgt_flat    = tgt_out.reshape(-1)                  # [B*(T-1)]
        loss = criterion(logits_flat, tgt_flat)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, pad_idx):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_inp, pad_idx)
            logits   = model(src, tgt_inp, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


# =============================================================================
# 5. INFERENCE (GREEDY DECODE)
# =============================================================================

def translate(sentence, model, src_vocab, tgt_vocab, max_len=MAX_LEN):
    """Dịch một câu từ Anh sang Việt (greedy decoding)."""
    model.eval()
    tgt_idx2word = {v: k for k, v in tgt_vocab.items()}

    # Encode source
    src_ids = torch.tensor([encode(sentence, src_vocab, max_len)],
                           dtype=torch.long).to(device)
    src_mask = make_src_mask(src_ids)

    with torch.no_grad():
        enc_out = model.encoder(src_ids, src_mask)   # [1, S, D]

        # Bắt đầu từ <BOS>
        tgt_ids = [tgt_vocab["<BOS>"]]
        for _ in range(max_len - 1):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            tgt_mask   = make_tgt_mask(tgt_tensor)

            dec_out = model.decoder(tgt_tensor, enc_out, tgt_mask, src_mask)
            logits  = model.out_proj(dec_out)   # [1, t, V]

            next_id = logits[0, -1].argmax(-1).item()
            tgt_ids.append(next_id)
            if next_id == tgt_vocab["<EOS>"]:
                break

    tokens = [tgt_idx2word.get(i, "<UNK>")
              for i in tgt_ids[1:]          # bỏ <BOS>
              if i not in (tgt_vocab["<PAD>"], tgt_vocab["<EOS>"])]
    return " ".join(tokens)


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────────
    pairs = load_pairs(SRC_FILE, TGT_FILE, n=N_SAMPLES)

    split = int(0.9 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    # ── Build vocabularies ─────────────────────────────────────────────────────
    src_vocab = build_vocab([s for s, _ in train_pairs], max_vocab=SRC_VOCAB)
    tgt_vocab = build_vocab([t for _, t in train_pairs], max_vocab=TGT_VOCAB)
    SRC_VOCAB = len(src_vocab)
    TGT_VOCAB = len(tgt_vocab)
    PAD_IDX   = src_vocab["<PAD>"]
    print(f"src_vocab: {SRC_VOCAB} | tgt_vocab: {TGT_VOCAB}")

    # ── Datasets & Loaders ─────────────────────────────────────────────────────
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,}")

    # ── Optimizer & Loss ───────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        t_loss = train_epoch(model, train_loader, optimizer, criterion, PAD_IDX)
        v_loss = eval_epoch(model, val_loader,   criterion, PAD_IDX)
        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f"  Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, "b-o", label="Train Loss")
    plt.plot(val_losses,   "r-o", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Transformer Decoder — En→Vi Translation")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    # ── Inference ──────────────────────────────────────────────────────────────
    print("\n--- Inference thử nghiệm ---")
    test_sentences = [
        "I love you.",
        "How are you?",
        "Tom went to school.",
        "She is very beautiful.",
        "Please close the door.",
    ]
    for sent in test_sentences:
        result = translate(sent, model, src_vocab, tgt_vocab)
        print(f"EN: {sent}")
        print(f"VI: {result}\n")
