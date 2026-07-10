import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Cấu hình & Constants
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2

# ==========================================
# 2. Xử lý Từ điển & Dữ liệu
# ==========================================
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def load_pairs(vi_path, en_path, num_samples=20000):
    if not os.path.exists(vi_path) or not os.path.exists(en_path):
        raise FileNotFoundError("Không tìm thấy tệp dữ liệu!")
        
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = f.read().split('\n')
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = f.read().split('\n')
    
    pairs = [[vi.strip(), en.strip()] for vi, en in zip(vi_lines[:num_samples], en_lines[:num_samples]) if vi and en]
    return pairs

def sentence_to_tensor(lang, sentence):
    indexes = [lang.word2index.get(word.lower(), 0) for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor([indexes], dtype=torch.long, device=device)

# ==========================================
# 3. Định nghĩa Mô hình
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        
    def forward(self, input):
        embedded = self.embedding(input)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.Wa = nn.Linear(hid_dim, hid_dim)
        self.Ua = nn.Linear(hid_dim, hid_dim)
        self.Va = nn.Linear(hid_dim, 1)

    def forward(self, query, values):
        query = query.transpose(0, 1) # (batch, 1, hid)
        score = self.Va(torch.tanh(self.Wa(query) + self.Ua(values)))
        weights = F.softmax(score, dim=1)
        context = torch.bmm(weights.transpose(1, 2), values)
        return context, weights

class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.attention = BahdanauAttention(hid_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, attn_weights

# ==========================================
# 4. Huấn luyện & Đánh giá
# ==========================================
def evaluate_metrics(encoder, decoder, pairs, criterion, input_lang, output_lang):
    encoder.eval(); decoder.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    
    with torch.no_grad():
        for pair in pairs:
            input_tensor = sentence_to_tensor(input_lang, pair[0])
            target_tensor = sentence_to_tensor(output_lang, pair[1])
            
            enc_outs, hidden = encoder(input_tensor)
            dec_in = torch.tensor([[SOS_token]], device=device)
            
            for t in range(target_tensor.size(1)):
                out, hidden, _ = decoder(dec_in, hidden, enc_outs)
                total_loss += criterion(out, target_tensor[:, t]).item()
                if out.argmax(1).item() == target_tensor[:, t].item():
                    total_correct += 1
                total_tokens += 1
                dec_in = target_tensor[:, t].unsqueeze(1)
                
    return total_loss/total_tokens, total_correct/total_tokens

def train_one_epoch(encoder, decoder, pairs, enc_opt, dec_opt, criterion, input_lang, output_lang):
    encoder.train(); decoder.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    
    pbar = tqdm(pairs, desc="Training")
    for pair in pbar:
        input_tensor = sentence_to_tensor(input_lang, pair[0])
        target_tensor = sentence_to_tensor(output_lang, pair[1])
        
        enc_opt.zero_grad(); dec_opt.zero_grad()
        enc_outs, hidden = encoder(input_tensor)
        dec_in = torch.tensor([[SOS_token]], device=device)
        
        loss_pair = 0
        for t in range(target_tensor.size(1)):
            out, hidden, _ = decoder(dec_in, hidden, enc_outs)
            loss_pair += criterion(out, target_tensor[:, t])
            if out.argmax(1).item() == target_tensor[:, t].item():
                total_correct += 1
            total_tokens += 1
            dec_in = target_tensor[:, t].unsqueeze(1)
            
        loss_pair.backward()
        enc_opt.step(); dec_opt.step()
        total_loss += loss_pair.item()
        pbar.set_postfix(loss=f"{loss_pair.item()/target_tensor.size(1):.4f}")
        
    return total_loss/total_tokens, total_correct/total_tokens

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    VI_PATH = r'd:\dut_ai\AIO_code\Attention\data\archive\vi_sents'
    EN_PATH = r'd:\dut_ai\AIO_code\Attention\data\archive\en_sents'
    
    # Load data
    raw_pairs = load_pairs(VI_PATH, EN_PATH, num_samples=10000)
    input_lang, output_lang = Vocabulary('vi'), Vocabulary('en')
    for p in raw_pairs:
        input_lang.addSentence(p[0]); output_lang.addSentence(p[1])
        
    train_pairs, val_pairs = train_test_split(raw_pairs, test_size=0.2, random_state=42)
    
    # Init Models
    encoder = Encoder(input_lang.n_words, 128, 256).to(device)
    decoder = AttnDecoder(output_lang.n_words, 128, 256).to(device)
    
    enc_opt = optim.Adam(encoder.parameters(), lr=0.001)
    dec_opt = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    
    # Train Loop
    for epoch in range(1, 4):
        t_loss, t_acc = train_one_epoch(encoder, decoder, train_pairs, enc_opt, dec_opt, criterion, input_lang, output_lang)
        v_loss, v_acc = evaluate_metrics(encoder, decoder, val_pairs, criterion, input_lang, output_lang)
        print(f"\nEpoch {epoch} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}\n")
