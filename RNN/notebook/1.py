import math
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        # Trọng số ánh xạ từ input (hoặc output của layer dưới) vào hidden state
        self.i2h = nn.Linear(input_size, hidden_size) 
        # Trọng số ánh xạ từ hidden state trước đó (t-1) vào hidden state hiện tại (t)
        self.h2h = nn.Linear(hidden_size, hidden_size) 
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

        # Dùng ModuleList để chứa các layer (stack) của RNN
        self.cells = nn.ModuleList() 
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(RNNCell(layer_input_size, hidden_size))

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        if hidden is None:   
            # Nếu không khởi tạo hidden state ban đầu, tự động gán bằng 0
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
        outputs = []
        # Tách hidden state ban đầu ra thành list chứa state của từng layer
        current_hidden_states = [hidden[i] for i in range(self.num_layers)] 
        
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
        # outputs shape: (batch_size, seq_length, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        
        # Gom các hidden state cuối cùng của tất cả các layer lại
        # final_hidden shape: (num_layers, batch_size, hidden_size)
        final_hidden = torch.stack(current_hidden_states, dim=0)
        
        return outputs, final_hidden

if __name__ == "__main__":
    batch_size = 32
    seq_length = 10
    input_size = 50   # Số chiều vector đầu vào (ví dụ: kích thước của Word Embedding)
    hidden_size = 64  # Số chiều vector ngữ cảnh (context/hidden vector)
    num_layers = 2    # Số stack RNN (2 layer)
  
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    
    # Tạo dữ liệu giả lập (batch_first = True)
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    
    # Chạy qua mô hình
    out, h_n = rnn(dummy_input)
    
    print(f"Output shape (batch_size, seq_length, hidden_size): {out.shape}")
    print(f"Hidden state cuối shape (num_layers, batch_size, hidden_size): {h_n.shape}")
