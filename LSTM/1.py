import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        # W_ih * x + W_hh * h_prev + bias
        gates = self.W_ih(x) + self.W_hh(h_prev) # Shape: (batch_size, 4 * hidden_size)

        # Cắt tensor thành 4 phần bằng nhau tương ứng với 4 cổng
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Đi qua các hàm kích hoạt (Activation functions)
        i_t = torch.sigmoid(i_gate)   # Input gate
        f_t = torch.sigmoid(f_gate)   # Forget gate
        g_t = torch.tanh(g_gate)      # Cell/Candidate gate
        o_t = torch.sigmoid(o_gate)   # Output gate

        # Cập nhật Cell State (c_t) và Hidden State (h_t) hiện tại
        c_t = (f_t * c_prev) + (i_t * g_t)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)

    def forward(self, x, hidden=None):
        # Chuyển x thành dạng (seq_length, batch_size, input_size) để vòng lặp for dễ cắt theo time-step
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_length, batch_size, _ = x.size()

        # Khởi tạo hidden state và cell state ban đầu bằng 0 nếu user không truyền vào
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = hidden

        outputs = []

        for t in range(seq_length):
            x_t = x[t] # Trích xuất dữ liệu tại thời điểm t. Shape: (batch_size, input_size)

            # Đưa qua lõi LSTM
            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))

            # Lưu lại hidden state để làm output tổng
            outputs.append(h_t)

        # Gom danh sách outputs thành tensor có dạng (seq_length, batch_size, hidden_size)
        outputs = torch.stack(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, (h_t, c_t)

def test_model():
    batch_size = 32
    seq_length = 10
    input_size = 50
    hidden_size = 128

    x = torch.randn(batch_size, seq_length, input_size)

    model = CustomLSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    # Cho dữ liệu đi qua mô hình
    outputs, (h_n, c_n) = model(x)

    print(f"{'Thông số':<25} | {'Kích thước (Shape)'}")
    print("-" * 55)
    print(f"{'Input (x)':<25} | {list(x.shape)}")
    print(f"{'Outputs (All h_t)':<25} | {list(outputs.shape)}")
    print(f"{'Hidden State cuối (h_n)':<25} | {list(h_n.shape)}")
    print(f"{'Cell State cuối (c_n)':<25} | {list(c_n.shape)}")

if __name__ == '__main__':
    test_model()
