import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import os

# ==========================================
# 1. Cấu hình & Constants
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

MAX_LEN_INPUT = 50
MAX_LEN_OUTPUT = 12

# ==========================================
# 2. Xử lý Từ điển
# ==========================================
class CharVocabulary:
    def __init__(self):
        self.char2idx = {"<PAD>": PAD_token, "<SOS>": SOS_token, "<EOS>": EOS_token, "<UNK>": UNK_token}
        self.idx2char = {PAD_token: "<PAD>", SOS_token: "<SOS>", EOS_token: "<EOS>", UNK_token: "<UNK>"}
        self.n_chars = 4

    def add_text(self, text):
        for char in text:
            if char not in self.char2idx:
                self.char2idx[char] = self.n_chars
                self.idx2char[self.n_chars] = char
                self.n_chars += 1

    def encode(self, text, max_len):
        tokens = [self.char2idx.get(c, UNK_token) for c in text]
        tokens = tokens[:max_len-1] + [EOS_token]
        tokens = tokens + [PAD_token] * (max_len - len(tokens))
        return tokens

# ==========================================
# 3. Các thành phần Seq2Seq
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)
            
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch, hid)
        # encoder_outputs: (batch, seq, hid)
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        h = hidden.transpose(0, 1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim):
        super().__init__()
        self.attention = Attention(hid_dim)
        self.embedding = nn.Embedding(output_dim, hid_dim)
        self.gru = nn.GRU(hid_dim * 2, hid_dim, batch_first=True)
        self.out = nn.Linear(hid_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        prediction = self.out(output.squeeze(1))
        return prediction, hidden

# --- Class Seq2Seq Tổng thể ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        # Token đầu tiên luôn là <SOS>
        input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            
            # Quyết định dùng Teacher Forcing hay dùng kết quả dự đoán trước đó
            top1 = output.argmax(1)
            input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1.unsqueeze(1)
            
        return outputs

# ==========================================
# 4. Huấn luyện
# ==========================================
def train_model(model, train_x, train_y, n_epochs=5, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        indices = np.arange(len(train_x))
        np.random.shuffle(indices)
        
        pbar = tqdm(range(0, len(train_x), batch_size), desc=f"Epoch {epoch}")
        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            src = torch.tensor(train_x[batch_idx]).long().to(device)
            trg = torch.tensor(train_y[batch_idx]).long().to(device)
            
            optimizer.zero_grad()
            output = model(src, trg) # (batch, trg_len, vocab)
            
            # Reshape để tính loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            if torch.isnan(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"-> Epoch {epoch} Loss: {epoch_loss/(len(train_x)/batch_size):.4f}")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    DATA_PATH = r'd:\dut_ai\AIO_code\Attention\data\data.csv'
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH, header=None).dropna()
    h_dates, m_dates = df[0].values, df[1].values
    
    in_vocab, out_vocab = CharVocabulary(), CharVocabulary()
    for h, m in zip(h_dates, m_dates):
        in_vocab.add_text(str(h).lower()); out_vocab.add_text(str(m))
        
    X = np.array([in_vocab.encode(str(h).lower(), MAX_LEN_INPUT) for h in h_dates])
    Y = np.array([out_vocab.encode(str(m), MAX_LEN_OUTPUT) for m in m_dates])
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. Init Seq2Seq Model
    encoder = Encoder(in_vocab.n_chars, 256).to(device)
    decoder = Decoder(out_vocab.n_chars, 256).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # 3. Train
    train_model(model, X_train, y_train, n_epochs=10)
    
    print("\n--- Huấn luyện Seq2Seq hoàn tất ---")
