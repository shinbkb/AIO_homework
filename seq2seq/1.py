

import random
import re

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
from pyvi import ViTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng device: {device}")

# Các token đặc biệt
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# Hyperparameters
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 64
N_EPOCHS = 100
PATIENCE = 5
MIN_DELTA = 0.001
CLIP = 1
LR = 0.001
NUM_SAMPLES = 20000



class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: SOS_TOKEN, 2: EOS_TOKEN, 3: UNK_TOKEN}
        self.stoi = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi[token] if token in self.stoi else self.stoi[UNK_TOKEN]
            for token in text
        ]


def tokenize_en(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text.split()


def tokenize_vi(text):
    text = text.lower().strip()
    return ViTokenizer.tokenize(text).split()


class TranslationDataset(Dataset):
    def __init__(self, en_path, vi_path, num_samples=30000):
        with open(en_path, "r", encoding="utf-8") as f:
            self.en_sentences = f.readlines()[:num_samples]
        with open(vi_path, "r", encoding="utf-8") as f:
            self.vi_sentences = f.readlines()[:num_samples]

        print(f"Đã nạp {len(self.en_sentences)} câu.")

        self.en_tokenized = [tokenize_en(sent) for sent in self.en_sentences]
        self.vi_tokenized = [tokenize_vi(sent) for sent in self.vi_sentences]

        self.en_vocab = Vocabulary(freq_threshold=2)
        self.en_vocab.build_vocabulary(self.en_tokenized)

        self.vi_vocab = Vocabulary(freq_threshold=2)
        self.vi_vocab.build_vocabulary(self.vi_tokenized)

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, index):
        en_num = (
            [self.en_vocab.stoi[SOS_TOKEN]]
            + self.en_vocab.numericalize(self.en_tokenized[index])
            + [self.en_vocab.stoi[EOS_TOKEN]]
        )
        vi_num = (
            [self.vi_vocab.stoi[SOS_TOKEN]]
            + self.vi_vocab.numericalize(self.vi_tokenized[index])
            + [self.vi_vocab.stoi[EOS_TOKEN]]
        )
        return torch.tensor(en_num), torch.tensor(vi_num)


def collate_fn(batch):
    en_batch, vi_batch = [], []
    for en_item, vi_item in batch:
        en_batch.append(en_item)
        vi_batch.append(vi_item)
    # shape: [seq_len, batch_size]
    en_batch = nn.utils.rnn.pad_sequence(en_batch, padding_value=0, batch_first=False)
    vi_batch = nn.utils.rnn.pad_sequence(vi_batch, padding_value=0, batch_first=False)
    return en_batch, vi_batch


# MÔ HÌNH

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]
        _, hidden = self.rnn(embedded)
        # hidden (Context Vector): [1, batch_size, hid_dim]
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)  # Thêm Context vector vào đầu vào
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input: [batch_size] -> thêm chiều seq_len = 1
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        emb_con = torch.cat((embedded, context), dim=2)  # [1, batch_size, emb_dim + hid_dim]
        output, hidden = self.rnn(emb_con, hidden)

        output = torch.cat(
            (embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1
        )
        prediction = self.fc_out(output)  # [batch_size, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size] | trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        context = self.encoder(src)
        hidden = context

        input = trg[0, :]  # Token <sos> ban đầu
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# CHIẾN LƯỢC DECODING

def translate_greedy(sentence, model, en_vocab, vi_vocab, device, max_len=50):
    """Dịch câu bằng Greedy Search (luôn chọn từ có xác suất cao nhất)."""
    model.eval()
    tokens = [SOS_TOKEN] + tokenize_en(sentence) + [EOS_TOKEN]
    src_indexes = [en_vocab.stoi.get(tok, en_vocab.stoi[UNK_TOKEN]) for tok in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        context = model.encoder(src_tensor)
        hidden = context

    trg_indexes = [vi_vocab.stoi[SOS_TOKEN]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, context)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == vi_vocab.stoi[EOS_TOKEN]:
            break

    trg_tokens = [vi_vocab.itos[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1])


def translate_sampling(sentence, model, en_vocab, vi_vocab, device, temperature=1.0, max_len=50):
    """Dịch câu bằng Temperature Sampling (lấy mẫu ngẫu nhiên từ phân phối xác suất)."""
    model.eval()
    tokens = [SOS_TOKEN] + tokenize_en(sentence) + [EOS_TOKEN]
    src_indexes = [en_vocab.stoi.get(tok, en_vocab.stoi[UNK_TOKEN]) for tok in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        context = model.encoder(src_tensor)
        hidden = context

    trg_indexes = [vi_vocab.stoi[SOS_TOKEN]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, context)

        probs = F.softmax(output.squeeze(0) / temperature, dim=0)
        pred_token = torch.multinomial(probs, 1).item()
        trg_indexes.append(pred_token)
        if pred_token == vi_vocab.stoi[EOS_TOKEN]:
            break

    trg_tokens = [vi_vocab.itos[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1])


def translate_beam_search(sentence, model, en_vocab, vi_vocab, device, beam_width=3, max_len=50):
    """Dịch câu bằng Beam Search với Length Penalty."""
    model.eval()
    tokens = [SOS_TOKEN] + tokenize_en(sentence) + [EOS_TOKEN]
    src_indexes = [en_vocab.stoi.get(tok, en_vocab.stoi[UNK_TOKEN]) for tok in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        context = model.encoder(src_tensor)
        hidden = context

    # Mỗi phần tử beam: (sequence, hidden_state, score)
    beam = [([vi_vocab.stoi[SOS_TOKEN]], hidden, 0.0)]
    completed_sentences = []

    for _ in range(max_len):
        candidates = []
        for seq, hid, score in beam:
            if seq[-1] == vi_vocab.stoi[EOS_TOKEN]:
                completed_sentences.append((seq, score))
                continue

            trg_tensor = torch.LongTensor([seq[-1]]).to(device)
            with torch.no_grad():
                output, new_hidden = model.decoder(trg_tensor, hid, context)

            # Log softmax để tránh underflow khi cộng dồn điểm
            log_probs = F.log_softmax(output.squeeze(0), dim=0)
            top_probs, top_idx = log_probs.topk(beam_width)
            for i in range(beam_width):
                candidates.append((
                    seq + [top_idx[i].item()],
                    new_hidden,
                    score + top_probs[i].item(),
                ))

        beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]

        if all(seq[-1] == vi_vocab.stoi[EOS_TOKEN] for seq, _, _ in beam):
            break

    if not completed_sentences:
        best_seq = beam[0][0]
    else:
        # Normalize score theo độ dài (Length Penalty)
        best_seq = max(completed_sentences, key=lambda x: x[1] / len(x[0]))[0]

    trg_tokens = [vi_vocab.itos[i] for i in best_seq]
    trg_tokens = [t for t in trg_tokens if t not in [SOS_TOKEN, EOS_TOKEN]]
    return " ".join(trg_tokens).replace("_", " ")


# HUẤN LUYỆN & ĐÁNH GIÁ

def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, total=len(iterator), desc=f"Epoch {epoch}", leave=False)
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def calculate_bleu_score(model, dataset, en_vocab, vi_vocab, device, num_samples=500):
    """Tính BLEU score trên một tập mẫu ngẫu nhiên."""
    model.eval()
    references, candidates = [], []
    test_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    print(f"Đang tính điểm BLEU trên {len(test_indices)} câu mẫu...")
    for i in tqdm(test_indices, leave=False):
        en_sentence = " ".join(dataset.en_tokenized[i])
        references.append([dataset.vi_tokenized[i]])
        translated = translate_greedy(en_sentence, model, en_vocab, vi_vocab, device)
        candidates.append(translated.split())
    score = corpus_bleu(references, candidates)
    return score * 100  # Thang điểm 100



# MAIN

def main():
    # --- Dữ liệu ---
    en_file = "en_sents"
    vi_file = "vi_sents"
    try:
        dataset = TranslationDataset(en_file, vi_file, num_samples=NUM_SAMPLES)
    except FileNotFoundError:
        print("Vui lòng sửa lại đường dẫn file English và Vietnamese.")
        return

    print("Vocab Eng size:", len(dataset.en_vocab))
    print("Vocab Vie size:", len(dataset.vi_vocab))

    # --- Tách train / validation ---
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Số batch Train: {len(train_loader)} | Valid: {len(valid_loader)}")

    # --- Khởi tạo mô hình ---
    INPUT_DIM = len(dataset.en_vocab)
    OUTPUT_DIM = len(dataset.vi_vocab)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    TRG_PAD_IDX = dataset.vi_vocab.stoi[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # --- Huấn luyện với Early Stopping ---
    best_valid_loss = float("inf")
    counter = 0
    print(f"Bắt đầu huấn luyện (Max {N_EPOCHS} epochs, Patience: {PATIENCE})...")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, epoch)
        valid_loss = evaluate(model, valid_loader, criterion)
        print(f"Epoch: {epoch:02} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}")

        if valid_loss < (best_valid_loss - MIN_DELTA):
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_model.pt")
            counter = 0
            print("  --> Cải thiện! Đã lưu mô hình.")
        else:
            counter += 1
            print(f"  --> Valid Loss không cải thiện ({counter}/{PATIENCE})")
            if counter >= PATIENCE:
                print(f"!!! EARLY STOPPING tại Epoch {epoch} !!!")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    print("Đã tải lại trọng số tối ưu nhất.")

    # --- BLEU Score ---
    nltk.download("punkt", quiet=True)
    bleu_score = calculate_bleu_score(
        model, dataset, dataset.en_vocab, dataset.vi_vocab, device, num_samples=300
    )
    print(f"\n🏆 Điểm BLEU: {bleu_score:.2f} / 100")
    print("(BLEU > 30: có thể hiểu được | > 50: chất lượng cao)")

    # --- So sánh các chiến lược decoding ---
    test_sentences = [
        "hello, how are you today?",
        "i love learning machine learning.",
        "the cat is sleeping on the table.",
    ]
    print("\n=== SO SÁNH CÁC CHIẾN LƯỢC DECODING ===\n")
    for sentence in test_sentences:
        print(f"Câu gốc (EN): {sentence}")
        greedy_res = translate_greedy(sentence, model, dataset.en_vocab, dataset.vi_vocab, device)
        print(f"-> Greedy:    {greedy_res.replace('_', ' ')}")
        beam_res = translate_beam_search(sentence, model, dataset.en_vocab, dataset.vi_vocab, device, beam_width=3)
        print(f"-> Beam (k=3): {beam_res}")
        samp_res = translate_sampling(sentence, model, dataset.en_vocab, dataset.vi_vocab, device, temperature=0.8)
        print(f"-> Sampling:  {samp_res.replace('_', ' ')}")
        print("-" * 50)


if __name__ == "__main__":
    main()
