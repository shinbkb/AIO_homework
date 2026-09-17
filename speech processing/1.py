# ============================================================
# Bài tập Thực hành 1: Nền tảng Xử lý Tín hiệu Âm thanh
# ============================================================
# Trong bài tập này, các bạn sẽ làm quen với việc đọc, phát,
# trực quan hóa và trích xuất các đặc trưng cơ bản của dữ liệu
# âm thanh. Đây là những bước tiền xử lý cốt lõi trước khi đưa
# dữ liệu vào các mô hình học sâu cho các bài toán nhận dạng
# giọng nói (Speech-to-Text), phân loại âm thanh, v.v.
# ============================================================
# Cài thư viện (chạy 1 lần trong terminal):
#   pip install librosa numpy matplotlib
# ============================================================

# ── 1. Import các thư viện cần thiết ──────────────────────────
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# IPython.display chỉ hoạt động trong Jupyter; trong .py dùng
# sounddevice hoặc bỏ qua phần phát âm thanh.
try:
    import IPython.display as ipd
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# ── 2. Tải dữ liệu âm thanh ────────────────────────────────────
# TODO 1: Tải file âm thanh
# Sử dụng librosa.load() để đọc file âm thanh.
# Gợi ý: có thể dùng librosa.example('trumpet') để dùng file mẫu có sẵn.

audio_path = librosa.example("trumpet")

# y  : mảng numpy chứa dạng sóng (waveform)
# sr : sample rate – số mẫu/giây (Hz)
y, sr = librosa.load(audio_path)

print(f"Sampling rate (Tần số lấy mẫu): {sr} Hz")
print(f"Độ dài mảng dữ liệu (số lượng samples): {len(y)}")
print(f"Thời lượng (giây): {len(y) / sr:.4f}")

# ── TODO 2: Phát âm thanh ──────────────────────────────────────
# Chỉ hoạt động trong Jupyter Notebook / JupyterLab
if IN_NOTEBOOK:
    display(ipd.Audio(y, rate=sr))  # noqa: F821

# ── 3. Trực quan hóa Dạng sóng (Waveform) ─────────────────────
# Waveform biểu diễn biên độ (amplitude) của tín hiệu âm thanh
# theo thời gian. Giúp nhìn nhận tổng quan về cấu trúc âm thanh
# (đoạn nào nói to, đoạn nào ngắt quãng).

# TODO 3: Vẽ biểu đồ Waveform
plt.figure(figsize=(14, 5))

librosa.display.waveshow(y, sr=sr, color="blue")

plt.title("Waveform")
plt.xlabel("Thời gian (s)")
plt.ylabel("Biên độ")
plt.tight_layout()
plt.show()

# ── 4. Biến đổi Fourier (FFT) ─────────────────────────────────
# Biến đổi Fourier chuyển tín hiệu từ miền thời gian (Time Domain)
# sang miền tần số (Frequency Domain). Qua đó biết âm thanh được
# cấu thành từ những tần số nào và biên độ của từng tần số.

# TODO 4: Tính toán và hiển thị Phổ biên độ (FFT)
# Bước 1: np.fft.rfft()  – tính FFT cho tín hiệu thực
# Bước 2: np.abs()       – tính biên độ (Magnitude)
# Bước 3: np.fft.rfftfreq(n, d=1/sr) – tính giá trị tần số trục X

fft_result = np.fft.rfft(y)
magnitude  = np.abs(fft_result)
freqs      = np.fft.rfftfreq(len(y), d=1 / sr)

plt.figure(figsize=(14, 5))
plt.plot(freqs, magnitude, color="red")
plt.title("Phổ biên độ (Magnitude Spectrum - FFT)")
plt.xlabel("Tần số (Hz)")
plt.ylabel("Biên độ (Magnitude)")
plt.grid(True)
plt.tight_layout()
plt.show()
