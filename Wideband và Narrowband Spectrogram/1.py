from pathlib import Path
import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np


def milliseconds_to_samples(milliseconds: float, sr: int) -> int:
    """Chuyển đổi thời lượng cửa sổ (ms) sang số lượng mẫu."""
    if milliseconds <= 0 or sr <= 0:
        raise ValueError("milliseconds và sr phải lớn hơn 0.")
    return max(1, int(round(milliseconds * sr / 1000.0)))


def framing(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Chia tín hiệu thành các khung liên tiếp có độ dài frame_length."""
    if len(signal) < frame_length:
        return np.empty((0, frame_length), dtype=signal.dtype)
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.empty((num_frames, frame_length), dtype=signal.dtype)
    for frame_index in range(num_frames):
        start = frame_index * hop_length
        frames[frame_index] = signal[start : start + frame_length]
    return frames


def my_stft(
    signal: np.ndarray, n_fft: int, hop_length: int, win_length: int
) -> np.ndarray:
    """Tính biến đổi Fourier thời gian ngắn (STFT) sử dụng cửa sổ Hanning."""
    frames = framing(signal, win_length, hop_length)
    window = np.hanning(win_length)
    windowed_frames = frames * window[np.newaxis, :]
    return np.fft.rfft(windowed_frames, n=n_fft, axis=1).T


def extract_db_spectrogram(
    signal: np.ndarray, sr: int, win_length: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trích xuất ma trận dB power spectrogram cùng trục tần số và thời gian."""
    if signal.ndim != 1:
        raise ValueError("Tín hiệu phải là mảng 1D.")
    if win_length <= 0 or hop_length <= 0:
        raise ValueError("win_length và hop_length phải lớn hơn 0.")

    # Tính kích thước n_fft là lũy thừa nhỏ nhất của 2 >= win_length
    n_fft = 1 << (win_length - 1).bit_length()
    stft_matrix = my_stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    power = np.maximum(np.abs(stft_matrix) ** 2, 1e-12)
    spectrogram_db = 10.0 * np.log10(power)
    spectrogram_db -= spectrogram_db.max()

    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    times = (
        np.arange(spectrogram_db.shape[1]) * hop_length + win_length / 2.0
    ) / sr
    return spectrogram_db, frequencies, times


def resolve_audio_path(custom_path: str | None = None) -> Path:
    """Xác định đường dẫn file âm thanh hợp lệ."""
    if custom_path:
        path = Path(custom_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    candidates = [
        Path("audio/speech.wav"),
        Path.cwd() / "Wideband và Narrowband Spectrogram" / "audio" / "speech.wav",
        Path.cwd() / "audio" / "speech.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "Không tìm thấy file âm thanh. Vui lòng đặt file tại './audio/speech.wav' "
        "hoặc chỉ định đường dẫn qua tham số '--audio'."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Khảo sát Wideband và Narrowband Spectrogram."
    )
    parser.add_argument(
        "--audio", type=str, default=None, help="Đường dẫn đến file speech.wav"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Đường dẫn lưu hình ảnh kết quả (nếu không đặt sẽ hiện cửa sổ)",
    )
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio)
    print(f"[+] Audio path: {audio_path}")

    # Đọc tín hiệu âm thanh chuẩn hóa mono 16 kHz
    sr_target = 16000
    speech_signal, speech_sr = librosa.load(audio_path, sr=sr_target, mono=True)
    speech_signal = speech_signal.astype(np.float64)
    duration = len(speech_signal) / speech_sr

    print(f"[+] Sample rate : {speech_sr} Hz")
    print(f"[+] Samples     : {len(speech_signal)}")
    print(f"[+] Duration    : {duration:.2f} s")

    # Cấu hình cửa sổ Wideband (5 ms) và Narrowband (50 ms)
    wideband_win_length = milliseconds_to_samples(5.0, speech_sr)
    wideband_hop_length = milliseconds_to_samples(2.5, speech_sr)

    narrowband_win_length = milliseconds_to_samples(50.0, speech_sr)
    narrowband_hop_length = milliseconds_to_samples(10.0, speech_sr)

    # Trích xuất spectrogram
    wb_db, wb_freqs, wb_times = extract_db_spectrogram(
        speech_signal, speech_sr, wideband_win_length, wideband_hop_length
    )
    nb_db, nb_freqs, nb_times = extract_db_spectrogram(
        speech_signal, speech_sr, narrowband_win_length, narrowband_hop_length
    )

    print(f"[+] Wideband shape   : {wb_db.shape}")
    print(f"[+] Narrowband shape : {nb_db.shape}")

    # Trực quan hóa kết quả
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)

    # 1. Dạng sóng thời gian
    time_axis = np.arange(len(speech_signal)) / speech_sr
    axes[0].plot(time_axis, speech_signal, linewidth=0.7, color="#1f77b4")
    axes[0].set(title="Speech Waveform", xlabel="Time (s)", ylabel="Amplitude")

    # 2. Wideband Spectrogram
    wb_img = axes[1].pcolormesh(
        wb_times, wb_freqs, wb_db, shading="auto", cmap="magma"
    )
    axes[1].set(
        title="Wideband Spectrogram — window 5 ms (Tối ưu độ phân giải thời gian)",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        ylim=(0, 5000),
    )
    fig.colorbar(wb_img, ax=axes[1], label="dB")

    # 3. Narrowband Spectrogram
    nb_img = axes[2].pcolormesh(
        nb_times, nb_freqs, nb_db, shading="auto", cmap="magma"
    )
    axes[2].set(
        title="Narrowband Spectrogram — window 50 ms (Tối ưu độ phân giải tần số)",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
        ylim=(0, 5000),
    )
    fig.colorbar(nb_img, ax=axes[2], label="dB")

    if args.save:
        plt.savefig(args.save, dpi=300)
        print(f"[+] Đã lưu biểu đồ tại: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()