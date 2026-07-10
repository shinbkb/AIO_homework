import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ════════════════════════════════════════════════════════
# 1. POSITIONAL ENCODING
# ════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════
# 2. MULTI-HEAD ATTENTION
# ════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """
    Dùng chung cho cả 3 loại attention:
      - Encoder Self-Attention
      - Decoder Masked Self-Attention  (truyền tgt_mask)
      - Decoder Cross-Attention        (Q=dec, K/V=enc)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
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
        q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d)
        return self.W_o(out), attn


# ════════════════════════════════════════════════════════
# 3. FEED-FORWARD NETWORK
# ════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════
# 4. ENCODER LAYER & ENCODER
# ════════════════════════════════════════════════════════
class EncoderLayer(nn.Module):
    """2 sub-layer: Self-Attn + FFN"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.drop1(attn))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_len=512, dropout=0.1):
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


# ════════════════════════════════════════════════════════
# 5. DECODER LAYER & DECODER  ← PHẦN CHÍNH BÀI 1
# ════════════════════════════════════════════════════════
class DecoderLayer(nn.Module):
    """
    3 sub-layer:
      1. Masked Self-Attention  (Q=K=V=tgt, mask=causal)
      2. Cross-Attention        (Q=tgt, K=V=enc_out)
      3. Feed-Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Sub-layer 1
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        # Sub-layer 2
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2      = nn.LayerNorm(d_model)
        self.drop2      = nn.Dropout(dropout)
        # Sub-layer 3
        self.ffn   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # 1. Masked Self-Attention
        a1, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt   = self.norm1(tgt + self.drop1(a1))
        # 2. Cross-Attention: Q=decoder, K/V=encoder
        a2, _ = self.cross_attn(tgt, enc_out, enc_out, mask=src_mask)
        tgt   = self.norm2(tgt + self.drop2(a2))
        # 3. FFN
        tgt   = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        out = self.pos_enc(self.embedding(tgt))
        for layer in self.layers:
            out = layer(out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.norm(out)   # [B, T, D]


# ════════════════════════════════════════════════════════
# 6. SEQ2SEQ TRANSFORMER
# ════════════════════════════════════════════════════════
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
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.out_proj(dec_out)   # [B, T, tgt_vocab]


# ════════════════════════════════════════════════════════
# 7. MASK UTILITIES
# ════════════════════════════════════════════════════════
def make_src_mask(src, pad_idx=0):
    """[B, 1, 1, S]"""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx=0):
    """Causal mask + padding mask → [B, 1, T, T]"""
    T        = tgt.size(1)
    causal   = torch.tril(torch.ones(T, T, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    return causal & pad_mask


# ════════════════════════════════════════════════════════
# 8. SANITY CHECK
# ════════════════════════════════════════════════════════
D_MODEL=128; NUM_HEADS=4; NUM_LAYERS=2; D_FF=256
SRC_VOCAB=500; TGT_VOCAB=500; MAX_LEN=32

src = torch.randint(1, SRC_VOCAB, (4, 20)).to(device)
tgt = torch.randint(1, TGT_VOCAB, (4, 15)).to(device)

model = Seq2SeqTransformer(SRC_VOCAB, TGT_VOCAB, D_MODEL,
                            NUM_LAYERS, NUM_HEADS, D_FF, MAX_LEN).to(device)

logits = model(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
print(f"Output shape: {logits.shape}")   # [4, 15, 500]
print(f"Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("✅ Bài 1 OK!")
