import os
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, data_dir, vocab=None, max_length=150):
        self.data_dir = data_dir
        self.max_length = max_length
        self.samples = []
        self.labels = []
        
        print(f"Đang đọc dữ liệu từ: {data_dir}...")
        # Đọc dữ liệu từ folder pos (nhãn 1) và neg (nhãn 0)
        for label_type, label in [('pos', 1), ('neg', 0)]:
            dir_path = os.path.join(data_dir, label_type)
            if not os.path.exists(dir_path): 
                continue
                
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as f:
                        text = f.read().lower() # Chuyển text về in thường
                        words = re.findall(r'\b[a-z]+\b', text) # Chỉ lấy chữ cái
                        self.samples.append(words)
                        self.labels.append(label)
                        
        # Xây dựng bộ từ điển (Vocabulary) nếu ở tập train
        if vocab is None:
            all_words = [word for sample in self.samples for word in sample]
            word_counts = Counter(all_words)
            # Lấy 10000 từ xuất hiện nhiều nhất
            common_words = [word for word, count in word_counts.most_common(10000)]
            self.vocab = {word: idx + 2 for idx, word in enumerate(common_words)}
            self.vocab['<PAD>'] = 0 # Dùng để chèn vào câu ngắn
            self.vocab['<UNK>'] = 1 # Đại diện cho từ lạ không có trong từ điển
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words = self.samples[idx]
        # Chuyển từ thành số (index)
        numericalized = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]
        
        # Cắt câu nếu quá dài, bù <PAD> nếu quá ngắn để bằng max_length
        if len(numericalized) > self.max_length:
            numericalized = numericalized[:self.max_length]
        else:
            numericalized = numericalized + [self.vocab['<PAD>']] * (self.max_length - len(numericalized))
            
        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size) # Trọng số ánh xạ từ input (hoặc output của layer dưới) vào hidden state
        self.h2h = nn.Linear(hidden_size, hidden_size) # Trọng số ánh xạ từ hidden state trước đó (t-1) vào hidden state hiện tại (t)
        self.activation = nn.Tanh()

    def forward(self, x, hidden):
        h_new = self.activation(self.i2h(x) + self.h2h(hidden))
        return h_new

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList() # Dùng ModuleList để chứa các layer (stack) của RNN
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(RNNCell(layer_input_size, hidden_size))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if hidden is None:   # Nếu không khởi tạo hidden state ban đầu, tự động gán bằng 0
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
        outputs = []
        current_hidden_states = [hidden[i] for i in range(self.num_layers)] # Tách hidden state ban đầu ra thành list chứa state của từng layer
        for t in range(seq_length):
            # Lấy input tại bước thời gian t, shape: (batch_size, input_size)
            layer_input = x[:, t, :]
            
            # Đưa qua lần lượt các layer (stack từ dưới lên trên)
            for layer_idx in range(self.num_layers):
                h_prev = current_hidden_states[layer_idx]
                
                # Đưa qua ô RNN của layer tương ứng
                h_new = self.cells[layer_idx](layer_input, h_prev)
                
                # Cập nhật lại hidden state cho layer hiện tại
                current_hidden_states[layer_idx] = h_new
                
                # Output của layer này chính là input cho layer tiếp theo phía trên
                layer_input = h_new
                
            # `layer_input` lúc này chính là output của layer trên cùng (top-most) tại time step t
            outputs.append(layer_input.unsqueeze(1))
            
        # Nối tất cả các output của layer cuối cùng theo chiều seq_length
        outputs = torch.cat(outputs, dim=1)
        
        # Gom các hidden state cuối cùng của tất cả các layer lại
        final_hidden = torch.stack(current_hidden_states, dim=0)
        
        return outputs, final_hidden

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(SentimentClassifier, self).__init__()
        # 1. Lớp Embedding: Dịch các số (index) thành vector (embed_size chiều)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. Lớp RNN (chính là class RNN)
        self.rnn = RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        
        # 3. Lớp Linear: Đưa context vector thành xác suất (1 chiều)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid() # Đưa output về khoảng [0, 1]

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x) # Biến thành: (batch_size, seq_length, embed_size)
        
        # Chạy qua RNN
        rnn_out, final_hidden = self.rnn(embeds)
        
        # Chỉ lấy state của LAYER CUỐI CÙNG làm đại diện ngữ cảnh cho toàn câu
        avg_hidden = rnn_out.mean(dim=1) 
        # Phân loại và đưa qua Sigmoid
        out = self.fc(avg_hidden)
        return self.sigmoid(out).squeeze()

if __name__ == "__main__":
    # Đường dẫn đến data của bạn
    train_dir = r"d:\dut_ai\AIO_code\Dence Representation\data\data_train\train"
    val_dir = r"d:\dut_ai\AIO_code\Dence Representation\data\data_train\test"
    final_test_dir = r"d:\dut_ai\AIO_code\Dence Representation\data\data_test\test"

    # Tạo Dataset và DataLoader
    train_dataset = SentimentDataset(train_dir, max_length=150)
    vocab = train_dataset.vocab # Dùng chung vocab cho tập test
    val_dataset = SentimentDataset(val_dir, vocab=vocab, max_length=150)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Xong phần load data!")

    # Các siêu tham số
    vocab_size = len(vocab)
    embed_size = 50
    hidden_size = 64
    num_layers = 1 # Dùng 1 layer cho CPU chạy nhanh hơn
    epochs = 50
    learning_rate = 0.001
    patience = 3 # Nếu sau 3 Epochs liên tiếp mà Val Loss không giảm thì dừng
    patience_counter = 0
    best_val_loss = float('inf') # Giá trị loss tốt nhất (khởi tạo là vô cực)
    best_model_path = "best_rnn_model.pth"
    
    # Thiết lập GPU hoặc CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Đang huấn luyện trên:", device)

    # Khởi tạo model
    model = SentimentClassifier(vocab_size, embed_size, hidden_size, num_layers).to(device)

    # Hàm Loss và Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Bắt đầu quá trình huấn luyện...")
    for epoch in range(epochs):
        # ==========================================
        # 1. GIAI ĐOẠN HUẤN LUYỆN (TRAINING)
        # ==========================================
        model.train() # Bật chế độ train
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_acc = train_correct / train_total
        
        # ==========================================
        # 2. GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION)
        # ==========================================
        model.eval() # Tắt tính năng tự động cập nhật trọng số
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # In kết quả tổng hợp của Epoch
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")
              
        if avg_val_loss < best_val_loss:
            # Nếu mô hình tốt lên -> Cập nhật kỷ lục và lưu lại trọng số
            print(f"   => Val Loss giảm từ {best_val_loss:.4f} xuống {avg_val_loss:.4f}. Đang lưu mô hình...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0 # Trả bộ đếm về 0
        else:
            # Nếu mô hình không tốt lên
            patience_counter += 1
            print(f"   => Cảnh báo: Val Loss không giảm. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n[!] EARLY STOPPING KÍCH HOẠT! Mô hình đã ngừng học thêm sau {epoch+1} Epochs.")
                break

    # Sau khi Train xong toàn bộ, load lại mô hình tốt nhất
    print("\nĐã load lại trọng số của Epoch tốt nhất để dùng cho tập Data_Test!")
    model.load_state_dict(torch.load(best_model_path))
    
    # ==========================================
    # 3. ĐÁNH GIÁ TRÊN TẬP FINAL TEST
    # ==========================================
    final_test_dataset = SentimentDataset(final_test_dir, vocab=vocab, max_length=150)
    final_test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)

    model.eval() 
    test_correct = 0
    test_total = 0

    print("Đang chấm điểm trên tập data_test (Final Test)...")
    with torch.no_grad():
        for inputs, labels in final_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()
            
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    final_acc = test_correct / test_total
    print(f"=> ĐIỂM CHÍNH THỨC TRÊN TẬP FINAL TEST: {final_acc * 100:.2f}%")
