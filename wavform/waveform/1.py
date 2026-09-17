import argparse
import math
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


class LibrispeechSubset(Dataset):
    """Dataset tiền xử lý, loại bỏ khoảng lặng và chia chunk âm thanh."""

    def __init__(
        self,
        data_path: Path | str,
        chunk_length_ms: float = 200.0,
        overlap_ms: float = 10.0,
        sample_rate: int = 16000,
        speaker_to_idx: dict[str, int] | None = None,
    ):
        self.data_path = Path(data_path)
        self.sample_rate = sample_rate
        self.chunk_length_samples = int(sample_rate * (chunk_length_ms / 1000.0))
        self.overlap_samples = int(sample_rate * (overlap_ms / 1000.0))
        self.step_samples = self.chunk_length_samples - self.overlap_samples

        self.data: list[torch.Tensor] = []
        self.labels: list[int] = []

        audio_files = sorted(
            list(self.data_path.rglob("*.flac")) + list(self.data_path.rglob("*.wav"))
        )

        # Trích xuất speaker ID từ đường dẫn
        if speaker_to_idx is None:
            speaker_ids = sorted(
                list(
                    set(
                        f.parent.parent.name
                        if f.parent.parent.name != self.data_path.name
                        else f.parent.name
                        for f in audio_files
                    )
                )
            )
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_ids)}
        else:
            self.speaker_to_idx = speaker_to_idx

        for audio_file in audio_files:
            spk_id = (
                audio_file.parent.parent.name
                if audio_file.parent.parent.name != self.data_path.name
                else audio_file.parent.name
            )
            if spk_id not in self.speaker_to_idx:
                continue
            label = self.speaker_to_idx[spk_id]

            # Đọc tín hiệu, loại bỏ khoảng lặng và chuẩn hóa biên độ
            wav, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            wav, _ = librosa.effects.trim(wav, top_db=30)
            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val

            # Chia frame dạng chunking
            total_samples = len(wav)
            if total_samples >= self.chunk_length_samples:
                for start in range(
                    0, total_samples - self.chunk_length_samples + 1, self.step_samples
                ):
                    chunk = wav[start : start + self.chunk_length_samples]
                    self.data.append(
                        torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                    )
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


class SincConv1d(nn.Module):
    """Lớp tích chập tham số hóa SincNet (Band-pass filterbank)."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Khởi tạo dải tần số theo thang Mel
        low_hz = 30.0
        high_hz = self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)

        mel_low = 2595.0 * np.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * np.log10(1.0 + high_hz / 700.0)
        mel_points = np.linspace(mel_low, mel_high, self.out_channels + 1)
        freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Tham số học được f1 và f_diff
        self.f1 = nn.Parameter(torch.Tensor(freq_points[:-1]).view(-1, 1))
        self.f_diff = nn.Parameter(torch.Tensor(np.diff(freq_points)).view(-1, 1))

        # Cửa sổ đối xứng Hamming
        n = torch.linspace(
            0,
            (self.kernel_size - 1) / 2,
            steps=int((self.kernel_size - 1) / 2) + 1,
        )
        self.register_buffer(
            "window", 0.54 - 0.46 * torch.cos(2 * math.pi * n / self.kernel_size)
        )

        # Trục thời gian n (nửa bên phải đối xứng qua tâm)
        t_right = (
            torch.linspace(
                1,
                (self.kernel_size - 1) / 2,
                steps=int((self.kernel_size - 1) / 2),
            )
            / self.sample_rate
        )
        self.register_buffer("t_right", t_right.view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1_clamped = self.min_low_hz + torch.abs(self.f1)
        f_diff_clamped = self.min_band_hz + torch.abs(self.f_diff)
        f2_clamped = torch.clamp(
            f1_clamped + f_diff_clamped, max=self.sample_rate / 2.0
        )

        t = self.t_right
        band_pass_right = (
            torch.sin(2 * math.pi * f2_clamped * t)
            - torch.sin(2 * math.pi * f1_clamped * t)
        ) / (2 * math.pi * t)
        band_pass_center = 2 * (f2_clamped - f1_clamped)
        band_pass_left = torch.flip(band_pass_right, dims=[-1])
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=-1
        )

        full_window = torch.cat(
            [torch.flip(self.window[1:], dims=[0]), self.window]
        )
        filters = band_pass * full_window
        filters = filters / (2 * (f2_clamped - f1_clamped))
        filters = filters.unsqueeze(1)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class SpeakerIDNet(nn.Module):
    """Mô hình SincNet nhận diện người nói."""

    def __init__(self, num_classes: int):
        super().__init__()
        # SincConv Block
        self.sinc_conv = SincConv1d(
            out_channels=80, kernel_size=251, sample_rate=16000
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.norm1 = nn.InstanceNorm1d(80, affine=True)
        self.act1 = nn.LeakyReLU(0.2)

        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.norm2 = nn.InstanceNorm1d(60, affine=True)
        self.act2 = nn.LeakyReLU(0.2)

        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5)
        self.pool3 = nn.MaxPool1d(kernel_size=3)
        self.norm3 = nn.InstanceNorm1d(60, affine=True)
        self.act3 = nn.LeakyReLU(0.2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(60, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc1_act = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc2_act = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc3_act = nn.LeakyReLU(0.2)

        self.out = nn.Linear(2048, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.pool1(torch.abs(self.sinc_conv(x)))))
        x = self.act2(self.norm2(self.pool2(self.conv1(x))))
        x = self.act3(self.norm3(self.pool3(self.conv2(x))))

        x = self.adaptive_pool(x).squeeze(-1)

        x = self.fc1_act(self.bn1(self.fc1(x)))
        x = self.fc2_act(self.bn2(self.fc2(x)))
        x = self.fc3_act(self.bn3(self.fc3(x)))

        return self.log_softmax(self.out(x))


def evaluate_frame_level(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> float:
    """Tính tỷ lệ chính xác trên từng frame độc lập (200ms)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return (correct / total) * 100.0 if total > 0 else 0.0


def evaluate_sentence_level(
    model: nn.Module,
    full_sentence_wav_path: Path,
    device: torch.device,
    sample_rate: int = 16000,
    chunk_ms: float = 200.0,
    overlap_ms: float = 10.0,
) -> int | None:
    """Tính toán xác suất trung bình trên tất cả các chunk của câu thoại để biểu quyết."""
    model.eval()
    chunk_samples = int(sample_rate * (chunk_ms / 1000.0))
    step_samples = chunk_samples - int(sample_rate * (overlap_ms / 1000.0))

    wav, _ = librosa.load(full_sentence_wav_path, sr=sample_rate, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    max_val = np.max(np.abs(wav))
    if max_val > 0:
        wav = wav / max_val

    chunks = []
    for start in range(0, len(wav) - chunk_samples + 1, step_samples):
        chunk = wav[start : start + chunk_samples]
        chunks.append(torch.tensor(chunk, dtype=torch.float32).unsqueeze(0))

    if not chunks:
        return None

    batch_tensor = torch.stack(chunks).to(device)
    with torch.no_grad():
        log_probs = model(batch_tensor)
        probs = torch.exp(log_probs)
        avg_probs = torch.mean(probs, dim=0)
        predicted_id = torch.argmax(avg_probs).item()

    return predicted_id


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và đánh giá Speaker Identification với SincNet."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./audio",
        help="Đường dẫn đến thư mục tập huấn luyện",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Đường dẫn đến thư mục tập test (mặc định lấy theo data_dir)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Số lượng epoch huấn luyện"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Kích thước batch"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Tốc độ học (Learning Rate)"
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Đường dẫn lưu biểu đồ kết quả (.png)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_dir = Path(args.test_dir) if args.test_dir else data_dir

    print(f"[+] Đang tải dữ liệu từ: {data_dir.resolve()}")
    dataset = LibrispeechSubset(data_path=data_dir)

    if len(dataset) == 0:
        raise ValueError(
            f"Không tìm thấy dữ liệu âm thanh hợp lệ (.wav/.flac) trong '{data_dir.resolve()}'!"
        )

    num_classes = len(dataset.speaker_to_idx)
    print(f"[+] Số lượng chunks trích xuất: {len(dataset)}")
    print(f"[+] Số lượng người nói: {num_classes} ({dataset.speaker_to_idx})")

    # Chia Train / Validation (80% / 20%)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Thiết bị thực thi: {device}")

    model = SpeakerIDNet(num_classes=num_classes).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_accuracies = [], []

    print("[+] Bắt đầu quá trình huấn luyện...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Đánh giá trên tập val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = (correct / total) * 100.0 if total > 0 else 0.0
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] - "
            f"Train Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}%"
        )

    # Đánh giá trên tập test (Frame-level & Sentence-level)
    print("\n[+] Đang đánh giá trên tập Test...")
    test_dataset = LibrispeechSubset(
        data_path=test_dir, speaker_to_idx=dataset.speaker_to_idx
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    frame_acc = evaluate_frame_level(model, test_loader, device=device)
    print(f"=== Frame-level Test Accuracy: {frame_acc:.2f}% ===")

    test_audio_files = sorted(
        list(test_dir.rglob("*.flac")) + list(test_dir.rglob("*.wav"))
    )
    correct_sentences = 0
    total_sentences = 0

    for audio_file in test_audio_files:
        spk_id = (
            audio_file.parent.parent.name
            if audio_file.parent.parent.name != test_dir.name
            else audio_file.parent.name
        )
        if spk_id not in dataset.speaker_to_idx:
            continue

        true_label = dataset.speaker_to_idx[spk_id]
        pred_label = evaluate_sentence_level(
            model=model, full_sentence_wav_path=audio_file, device=device
        )

        if pred_label is not None:
            if pred_label == true_label:
                correct_sentences += 1
            total_sentences += 1

    sentence_acc = (
        (correct_sentences / total_sentences) * 100.0
        if total_sentences > 0
        else 0.0
    )
    print(
        f"=== Sentence-level Test Accuracy: {sentence_acc:.2f}% "
        f"({correct_sentences}/{total_sentences} files) ==="
    )

    # Trực quan hóa Loss / Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, marker="o", color="#1f77b4")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, args.epochs + 1), val_accuracies, marker="o", color="#ff7f0e"
    )
    plt.title("Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=300)
        print(f"[+] Đã lưu biểu đồ tại: {args.save_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()