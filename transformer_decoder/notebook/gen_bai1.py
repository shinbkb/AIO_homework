import nbformat

nb = nbformat.v4.new_notebook()
cells = []

# ── MARKDOWN: tiêu đề ─────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
# Bài 1: Transformer Decoder Nhiều Lớp
**Yêu cầu:** Code lại kiến trúc Transformer Decoder nhiều lớp **không dùng** `nn.TransformerDecoderLayer`.

## Kiến trúc tổng quan
```
Decoder Layer (x N lớp):
  ┌─────────────────────────────────────────┐
  │ 1. Masked Multi-Head Self-Attention     │  ← causal mask (không nhìn tương lai)
  │    + Add & LayerNorm                    │
  │ 2. Cross-Attention (Q=dec, K/V=enc)    │  ← chú ý vào encoder output
  │    + Add & LayerNorm                    │
  │ 3. Feed-Forward Network                │
  │    + Add & LayerNorm                    │
  └─────────────────────────────────────────┘
```
"""))

# ── CELL 1: imports ────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 1. Positional Encoding"))

# ── CELL 2: PositionalEncoding ────────────────────────────────────────────────
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
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, D]
        return self.dropout(x + self.pe[:, :x.size(1)])
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 2. Multi-Head Attention"))

# ── CELL 3: MultiHeadAttention ────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
class MultiHeadAttention(nn.Module):
    \"\"\"
    Scaled Dot-Product Multi-Head Attention.
    Dùng cho cả 3 loại: Self-Attn (Encoder), Masked Self-Attn (Decoder), Cross-Attn (Decoder).
    \"\"\"
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.h   = num_heads
        self.d_k = d_model // num_heads
        self.d   = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # Project & split heads → [B, h, L, d_k]
        q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        # Merge heads → [B, L, d_model]
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d)
        return self.W_o(out), attn
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 3. Position-wise Feed-Forward Network"))

# ── CELL 4: FFN ───────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
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
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 4. Encoder Layer & Transformer Encoder"))

# ── CELL 5: EncoderLayer + TransformerEncoder ─────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
class EncoderLayer(nn.Module):
    \"\"\"
    Encoder Layer gồm 2 sub-layer:
      1. Multi-Head Self-Attention
      2. Feed-Forward Network
    \"\"\"
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Sub-layer 1: Self-Attention
        attn, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop1(attn))
        # Sub-layer 2: FFN
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        out = self.pos_enc(self.embedding(src))
        for layer in self.layers:
            out = layer(out, src_mask)
        return self.norm(out)   # [B, S, D]
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 5. Decoder Layer & Transformer Decoder
> **Đây là phần chính của Bài 1.**

Khác với `EncoderLayer`, mỗi `DecoderLayer` có **3 sub-layer**:
| Sub-layer | Loại Attention | Q | K | V | Mask |
|---|---|---|---|---|---|
| 1 | Masked Self-Attention | tgt | tgt | tgt | Causal mask |
| 2 | Cross-Attention | tgt | enc_out | enc_out | src padding mask |
| 3 | Feed-Forward | — | — | — | — |
"""))

# ── CELL 6: DecoderLayer ──────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
class DecoderLayer(nn.Module):
    \"\"\"
    Decoder Layer gồm 3 sub-layer:
      1. Masked Multi-Head Self-Attention  (causal mask — không nhìn token tương lai)
      2. Cross-Attention                   (Q từ decoder, K/V từ encoder output)
      3. Position-wise Feed-Forward Network
    \"\"\"
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)

        # Sub-layer 2: Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)

        # Sub-layer 3: Feed-Forward
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention: Q=K=V=tgt (có causal mask)
        a1, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt   = self.norm1(tgt + self.drop1(a1))

        # 2. Cross-Attention: Q=tgt (từ decoder), K=V=enc_out (từ encoder)
        a2, _ = self.cross_attn(tgt, enc_out, enc_out, mask=src_mask)
        tgt   = self.norm2(tgt + self.drop2(a2))

        # 3. Feed-Forward
        tgt   = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt
"""))

# ── CELL 7: TransformerDecoder ────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
class TransformerDecoder(nn.Module):
    \"\"\"Stack N DecoderLayer.\"\"\"
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        out = self.pos_enc(self.embedding(tgt))   # [B, T, D]
        for layer in self.layers:
            out = layer(out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(out)   # [B, T, D]
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 6. Seq2Seq Transformer (Encoder + Decoder)"))

# ── CELL 8: Seq2SeqTransformer ────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
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
        enc_out = self.encoder(src, src_mask)                    # [B, S, D]
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask) # [B, T, D]
        return self.out_proj(dec_out)                            # [B, T, tgt_vocab]
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 7. Mask Utilities"))

# ── CELL 9: masks ─────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
def make_src_mask(src, pad_idx=0):
    \"\"\"Mask padding tokens trong source. Shape: [B, 1, 1, S]\"\"\"
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx=0):
    \"\"\"
    Kết hợp causal mask + padding mask cho target. Shape: [B, 1, T, T]
    - causal: vị trí i chỉ nhìn được 0..i (tam giác dưới)
    - padding: che các vị trí là PAD
    \"\"\"
    T        = tgt.size(1)
    causal   = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # [T, T]
    causal   = causal.unsqueeze(0).unsqueeze(0)                         # [1,1,T,T]
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)               # [B,1,1,T]
    return causal & pad_mask                                             # [B,1,T,T]
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("## 8. Kiểm tra kiến trúc (Sanity Check)"))

# ── CELL 10: sanity check ─────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# Hyperparameters demo
D_MODEL    = 128
NUM_HEADS  = 4
NUM_LAYERS = 2
D_FF       = 256
DROPOUT    = 0.1
SRC_VOCAB  = 500
TGT_VOCAB  = 500
MAX_LEN    = 32
BATCH      = 4
SRC_LEN    = 20
TGT_LEN    = 15

# Tạo dữ liệu giả
src = torch.randint(1, SRC_VOCAB, (BATCH, SRC_LEN)).to(device)
tgt = torch.randint(1, TGT_VOCAB, (BATCH, TGT_LEN)).to(device)

# Tạo masks
src_mask = make_src_mask(src)
tgt_mask = make_tgt_mask(tgt)
print(f"src_mask shape : {src_mask.shape}")  # [4, 1, 1, 20]
print(f"tgt_mask shape : {tgt_mask.shape}")  # [4, 1, 15, 15]

# Khởi tạo model
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
print(f"Total params   : {n_params:,}")

# Forward pass
logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
print(f"Output shape   : {logits.shape}")  # [4, 15, 500] ← [B, T, tgt_vocab]
print("\\n✅ Kiến trúc hoạt động đúng!")
"""))

# ── MARKDOWN ───────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
## 9. Tóm tắt

| Class | Mô tả |
|---|---|
| `PositionalEncoding` | Thêm thông tin vị trí vào embedding |
| `MultiHeadAttention` | Scaled dot-product attention với nhiều head |
| `PositionwiseFeedForward` | FFN gồm 2 Linear + ReLU |
| `EncoderLayer` | 2 sub-layer: Self-Attn + FFN |
| **`DecoderLayer`** | **3 sub-layer: Masked Self-Attn + Cross-Attn + FFN** |
| `TransformerEncoder` | Stack N `EncoderLayer` |
| `TransformerDecoder` | Stack N `DecoderLayer` |
| `Seq2SeqTransformer` | Ghép Encoder + Decoder + Linear |
| `make_src_mask` | Mask padding trong source |
| `make_tgt_mask` | Causal mask + padding mask cho target |
"""))

nb.cells = cells

import os
out_path = r"d:\dut_ai\AIO_code\transformer_decoder\notebook\1.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Done! {out_path}")
