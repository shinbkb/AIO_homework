import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import nltk
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# Global variables for vocabulary
word_to_idx = {"<unk>": 0, "<pad>": 1}
PAD_IDX = 1

def custom_tokenizer(text):
    return nltk.word_tokenize(text.lower())

def text_pipeline(text):
    tokens = custom_tokenizer(text)
    return [word_to_idx.get(token, word_to_idx["<unk>"]) for token in tokens]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_tensor = torch.tensor(text_pipeline(self.texts[idx]), dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_tensor, label_tensor

def collate_batch(batch):
    text_list, label_list = [], []
    for (_text, _label) in batch:
        text_list.append(_text)
        label_list.append(_label)

    # Padding các tensor trong cùng 1 batch để có kích thước bằng nhau
    text_list = pad_sequence(text_list, padding_value=PAD_IDX, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.long)
    return text_list, label_list

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

        # Nhân 2 vì sử dụng Bidirectional LSTM (2 chiều)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_length, embed_dim)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        # hidden có shape: (num_layers * num_directions, batch_size, hidden_dim)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]

        # Nối (concatenate) vector của 2 chiều lại với nhau
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)

        out = self.dropout(hidden_cat)
        out = self.fc(out) # (batch_size, num_classes)
        return out

def main():
    # 1. Tải và chuẩn bị dữ liệu
    try:
        df = pd.read_csv('sentiment_data.csv')
    except FileNotFoundError:
        print("Không tìm thấy file 'sentiment_data.csv'. Vui lòng kiểm tra lại.")
        return

    df.dropna(subset=['Comment', 'Sentiment'], inplace=True)

    texts = df['Comment'].astype(str).tolist()
    labels = df['Sentiment'].astype(int).tolist()

    # Split into training + validation (85%) and test sets (15%)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    # Split training + validation into final training (70%) and validation sets (15%)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=3/17, random_state=42, stratify=train_val_labels
    )

    print(f"Kích thước tập Train: {len(train_texts)}")
    print(f"Kích thước tập Validation: {len(val_texts)}")
    print(f"Kích thước tập Test: {len(test_texts)}")

    # 2. Xây dựng từ điển (Vocabulary)
    word_counts = Counter()
    for text in train_texts:
        word_counts.update(custom_tokenizer(text))

    global word_to_idx
    current_idx = 2
    for word, count in word_counts.most_common():
        word_to_idx[word] = current_idx
        current_idx += 1

    vocab_size = len(word_to_idx)
    print(f"Kích thước từ điển (Vocabulary size): {vocab_size}")

    # 3. Tạo Dataset và DataLoader
    BATCH_SIZE = 128
    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)
    test_dataset = SentimentDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # 4. Khởi tạo mô hình
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 3   # (0, 1, 2)
    NUM_LAYERS = 2

    model = SentimentModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS).to(device)
    print(model)

    # 5. Huấn luyện mô hình
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 50
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0

        train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1} Training")
        for texts, labels in train_loop:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(texts)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            acc = (predictions.argmax(1) == labels).sum().item()
            train_acc += acc

            train_loop.set_postfix(loss=train_loss/len(train_loader), acc=train_acc/len(train_dataset))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_dataset)

        model.eval()
        val_loss, val_acc = 0, 0

        val_loop = tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for texts, labels in val_loop:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts)

                loss = criterion(predictions, labels)
                val_loss += loss.item()
                acc = (predictions.argmax(1) == labels).sum().item()
                val_acc += acc

                val_loop.set_postfix(loss=val_loss/len(val_loader), acc=val_acc/len(val_dataset))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_dataset)

        print(f"Epoch: {epoch+1:02} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}% ")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_sentiment_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Dừng sớm tại Epoch {epoch+1} do validation loss không cải thiện sau {patience} epoch.")
                break

    # 6. Đánh giá trên tập Test
    model.eval()
    test_loss, test_acc = 0, 0

    test_loop = tqdm(test_loader, leave=False, desc="Đánh giá trên tập Test")
    with torch.no_grad():
        for texts, labels in test_loop:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)

            loss = criterion(predictions, labels)
            test_loss += loss.item()
            acc = (predictions.argmax(1) == labels).sum().item()
            test_acc += acc

            test_loop.set_postfix(loss=test_loss/len(test_loader), acc=test_acc/len(test_dataset))

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_dataset)

    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc*100:.2f}%")

if __name__ == '__main__':
    main()
