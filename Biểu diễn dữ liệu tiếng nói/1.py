from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

plt.style.use("seaborn-v0_8-whitegrid")
np.set_printoptions(precision=3, suppress=True)

# Đường dẫn đến file audio mẫu
AUDIO_PATH = Path("audio/speech.wav")


# ============================================================
# 1. CÁC HÀM CỐT LÕI TỰ CÀI ĐẶT
# ============================================================

def framing(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Chia tín hiệu thành các frame chồng lấp nhau (chỉ lấy full frames)."""
    num_frames = 1 + int((len(signal) - frame_length) / hop_length)
    frames = np.empty((num_frames, frame_length), dtype=signal.dtype)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = signal[start : start + frame_length]
    return frames


def my_stft(signal: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """Tính Short-Time Fourier Transform (STFT) thủ công sử dụng cửa sổ Hann."""
    frames = framing(signal, win_length, hop_length)
    window = np.hanning(win_length)
    windowed_frames = frames * window
    
    # rFFT dọc theo từng frame, zero-pad đến n_fft nếu n_fft > win_length
    stft_matrix = np.fft.rfft(windowed_frames, n=n_fft, axis=-1)
    
    # Trả về shape (n_fft // 2 + 1, num_frames) để khớp chuẩn Librosa
    return stft_matrix.T


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """Tính Magnitude Spectrogram từ STFT complex matrix."""
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """Tính Power Spectrogram (|X|^2) từ STFT complex matrix."""
    return np.abs(stft_matrix) ** 2


def power_to_db(power_spec: np.ndarray, top_db: float = 80.0) -> np.ndarray:
    """Chuyển đổi Power Spectrogram sang thang Decibel (dB) với dynamic range top_db."""
    ref_value = np.max(power_spec)
    eps = 1e-10
    power_db = 10.0 * np.log10(np.maximum(power_spec, eps) / np.maximum(ref_value, eps))
    power_db = np.maximum(power_db, power_db.max() - top_db)
    return power_db


# ============================================================
# 2. THỰC NGHIỆM TÍN HIỆU TỔNG HỢP (SYNTHETIC SIGNAL)
# ============================================================

def run_synthetic_experiments():
    print("--- Đang chạy thực nghiệm tín hiệu tổng hợp ---")
    synthetic_sr = 8000
    duration = 1.0
    n_samples = int(synthetic_sr * duration)
    t = np.arange(n_samples) / synthetic_sr

    tone_440 = 0.8 * np.sin(2 * np.pi * 440 * t)
    tone_880 = 0.8 * np.sin(2 * np.pi * 880 * t)
    synthetic_signal = np.concatenate([tone_440, tone_880])

    demo_frame_length = int(0.025 * synthetic_sr)  # 25 ms
    demo_hop_length = int(0.010 * synthetic_sr)    # 10 ms
    n_fft_demo = 256

    stft_demo = my_stft(synthetic_signal, n_fft_demo, demo_hop_length, demo_frame_length)
    power_demo = power_spectrogram(stft_demo)
    db_demo = power_to_db(power_demo)

    assert np.isfinite(db_demo).all(), "Lỗi: Giá trị dB chứa NaN hoặc Inf!"
    print(f"STFT demo shape: {stft_demo.shape} (Expected freq bins: {n_fft_demo // 2 + 1})")

    # So sánh Time-Frequency resolution với các kích thước window khác nhau
    window_sizes_ms = [10, 25, 50]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True, constrained_layout=True)

    for ax, win_ms in zip(axes, window_sizes_ms):
        w_len = round(win_ms / 1000 * synthetic_sr)
        h_len = max(1, w_len // 4)
        n_f = 1 << int(np.ceil(np.log2(w_len)))
        
        cur_stft = my_stft(synthetic_signal, n_f, h_len, w_len)
        cur_db = power_to_db(power_spectrogram(cur_stft))
        
        times = np.arange(cur_db.shape[1]) * h_len / synthetic_sr
        freqs = np.fft.rfftfreq(n_f, d=1 / synthetic_sr)
        
        img = ax.pcolormesh(times, freqs, cur_db, shading="auto", cmap="magma")
        ax.axvline(1.0, color="cyan", linestyle="--", linewidth=1)
        ax.set(title=f"Window = {win_ms} ms", xlabel="Time (s)", ylim=(300, 1100))

    axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=axes, label="dB", shrink=0.9)
    plt.show()


# ============================================================
# 3. THỰC NGHIỆM TRÊN TÍN HIỆU TIẾNG NÓI (SPEECH SIGNAL)
# ============================================================

def process_speech_pipeline(audio_path: Path):
    if not audio_path.exists():
        print(f"[Cảnh báo] Không tìm thấy file tại '{audio_path}'. Bỏ qua phần speech.")
        return

    print(f"\n--- Đang xử lý file âm thanh: {audio_path} ---")
    speech_signal, speech_sr = librosa.load(audio_path, sr=16000, mono=True)
    speech_signal = speech_signal.astype(np.float64)
    duration = len(speech_signal) / speech_sr
    print(f"Sample rate: {speech_sr} Hz | Duration: {duration:.2f} s | Samples: {len(speech_signal)}")

    # Thiết lập tham số phân tích STFT
    win_length = round(0.025 * speech_sr)  # 25 ms
    hop_length = round(0.010 * speech_sr)  # 10 ms
    n_fft = 1 << int(np.ceil(np.log2(win_length)))
    n_mels = 40

    # 1. STFT & Power Spectrogram tự cài đặt
    speech_stft = my_stft(speech_signal, n_fft, hop_length, win_length)
    speech_power = power_spectrogram(speech_stft)
    speech_db = power_to_db(speech_power)

    # 2. Sinh Mel-Spectrogram & Log-Mel qua Librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        S=speech_power,
        sr=speech_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 3. Sinh 13 hệ số MFCC từ Log-Mel qua Librosa
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        sr=speech_sr,
        n_mfcc=13
    )

    print("\nShape của các tầng biểu diễn:")
    print(f" - Power Spectrogram : {speech_power.shape}")
    print(f" - Mel-Spectrogram   : {mel_spectrogram.shape}")
    print(f" - Log-Mel           : {log_mel.shape}")
    print(f" - MFCC              : {mfcc.shape}")

    # Trực quan hóa so sánh 4 mức biểu diễn
    frame_times = np.arange(speech_power.shape[1]) * hop_length / speech_sr
    frequencies = np.fft.rfftfreq(n_fft, d=1 / speech_sr)
    time_axis = np.arange(len(speech_signal)) / speech_sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    # 1. Waveform
    axes[0].plot(time_axis, speech_signal, color="steelblue", linewidth=0.65)
    axes[0].set(title="1. Waveform — Tín hiệu 1D", xlabel="Time (s)", ylabel="Amplitude")

    # 2. Linear Power Spectrogram
    img1 = axes[1].pcolormesh(frame_times, frequencies, speech_db, shading="auto", cmap="magma")
    axes[1].set(title="2. Linear Power Spectrogram (dB)", xlabel="Time (s)", ylabel="Frequency (Hz)")
    fig.colorbar(img1, ax=axes[1], label="dB")

    # 3. Log-Mel Spectrogram
    img2 = axes[2].pcolormesh(frame_times, np.arange(n_mels), log_mel, shading="auto", cmap="magma")
    axes[2].set(title="3. Log-Mel Spectrogram", xlabel="Time (s)", ylabel="Mel bin")
    fig.colorbar(img2, ax=axes[2], label="dB")

    # 4. MFCC
    img3 = axes[3].pcolormesh(frame_times, np.arange(mfcc.shape[0]), mfcc, shading="auto", cmap="coolwarm")
    axes[3].set(title="4. MFCC (13 hệ số)", xlabel="Time (s)", ylabe0 l="Coefficient")
    fig.colorbar(img3, ax=axes[3], label="Value")

    plt.show()

    # 4. Đối chiếu STFT tự cài đặt với librosa.stft
    librosa_stft = librosa.stft(
        speech_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=False,
    )
    diff = np.max(np.abs(speech_stft - librosa_stft))
    print(f"\nSai số tuyệt đối cực đại giữa my_stft và librosa.stft: {diff:.6e}")


if __name__ == "__main__":
    run_synthetic_experiments()
    process_speech_pipeline(AUDIO_PATH)