"""Train a four-band CRNN with weighted score-level fusion on ESC-50.

Example:
    python train_multiband_crnn_esc50.py --data-root /path/to/ESC-50-master
    python train_multiband_crnn_esc50.py --download --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


SEED = 42
SAMPLE_RATE = 44_100
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_FFT = 1024
HOP_LENGTH = 441
SUB_N_MELS = 60
NUM_CLASSES = 50
SUB_BANDS = ((0, 3_000), (3_000, 6_000), (6_000, 10_000), (10_000, 22_050))
FUSION_WEIGHTS = (0.4, 0.2, 0.2, 0.2)
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_esc50(data_root: Path) -> None:
    """Download ESC-50 only if ``data_root`` does not already contain it."""
    if (data_root / "meta" / "esc50.csv").exists():
        print(f"Using existing dataset: {data_root.resolve()}")
        return
    if data_root.name != "ESC-50-master":
        raise ValueError("For --download, --data-root must end with 'ESC-50-master'.")

    archive = data_root.parent / "esc50_master.zip"
    data_root.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading ESC-50 (~600 MB)...")
    urllib.request.urlretrieve(ESC50_URL, archive)
    print("Extracting dataset...")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(data_root.parent)
    archive.unlink()


class ESC50Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.mel_transforms = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=SUB_N_MELS,
                f_min=float(low),
                f_max=float(high),
                power=2.0,
            )
            for low, high in SUB_BANDS
        ]
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _fix_length(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < NUM_SAMPLES:
            return F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[-1]))
        return waveform[..., :NUM_SAMPLES]

    def _build_sub_spectrograms(self, waveform: torch.Tensor) -> torch.Tensor:
        bands = []
        for mel_transform in self.mel_transforms:
            log_mel = self.db_transform(mel_transform(waveform)).squeeze(0)
            delta = torchaudio.functional.compute_deltas(log_mel)
            delta_delta = torchaudio.functional.compute_deltas(delta)
            features = torch.stack((log_mel, delta, delta_delta))
            mean = features.mean(dim=(1, 2), keepdim=True)
            std = features.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            bands.append((features - mean) / std)
        return torch.stack(bands).float()  # [4, 3, 60, time]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        waveform, sample_rate = torchaudio.load(self.audio_dir / row["filename"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
        return self._build_sub_spectrograms(self._fix_length(waveform)), int(row["target"])


class CRNNBranch(nn.Module):
    """One frequency-band branch: ResNet18 feature extractor followed by BiGRU."""

    def __init__(self, hidden_size: int = 256, gru_layers: int = 2) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False
        for parameter in backbone.layer4.parameters():
            parameter.requires_grad = True

        self.gru = nn.GRU(
            input_size=512 * 2,  # ResNet18 maps 60 Mel bins to 2 frequency bins.
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(inputs)  # [batch, channels, freq, time]
        batch, channels, freq, time_steps = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(batch, time_steps, channels * freq)
        _, hidden = self.gru(sequence)
        return self.classifier(torch.cat((hidden[-2], hidden[-1]), dim=1))


class MultiBandCRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(CRNNBranch() for _ in SUB_BANDS)
        self.register_buffer("fusion_weights", torch.tensor(FUSION_WEIGHTS))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack(
            [torch.softmax(branch(inputs[:, band]), dim=1) for band, branch in enumerate(self.branches)],
            dim=1,
        )
        fused = (probabilities * self.fusion_weights.view(1, -1, 1)).sum(dim=1)
        return fused.clamp_min(1e-8).log()  # Appropriate for NLLLoss.


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        log_probabilities = model(inputs)
        loss_sum += criterion(log_probabilities, targets).item() * inputs.size(0)
        correct += (log_probabilities.argmax(dim=1) == targets).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def train(
    model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, epochs: int,
) -> tuple[float, float]:
    best_accuracy = -1.0
    best_state = None
    started_at = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(inputs)
            loss = criterion(log_probabilities, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * inputs.size(0)
            correct += (log_probabilities.argmax(dim=1) == targets).sum().item()

        train_loss = loss_sum / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_accuracy, time.perf_counter() - started_at


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("ESC-50-master"))
    parser.add_argument("--download", action="store_true", help="Download ESC-50 if absent.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("multiband_crnn_esc50_best.pt"))
    args = parser.parse_args()

    if args.download:
        download_esc50(args.data_root)
    if not (args.data_root / "meta" / "esc50.csv").exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root}. Use --download or set --data-root.")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 2 if device.type == "cuda" else 0
    print(f"Device: {device}")

    metadata = pd.read_csv(args.data_root / "meta" / "esc50.csv")
    audio_dir = args.data_root / "audio"
    train_data = metadata[metadata["fold"].isin((1, 2, 3))]
    validation_data = metadata[metadata["fold"] == 4]
    test_data = metadata[metadata["fold"] == 5]
    print(f"Train: {len(train_data)} | Validation: {len(validation_data)} | Test: {len(test_data)}")

    loader_options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(ESC50Dataset(train_data, audio_dir), args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(ESC50Dataset(validation_data, audio_dir), args.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(ESC50Dataset(test_data, audio_dir), args.batch_size, shuffle=False, **loader_options)

    model = MultiBandCRNN().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    best_validation_accuracy, elapsed_seconds = train(model, train_loader, validation_loader, optimizer, criterion, device, args.epochs)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), args.output)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {elapsed_seconds:.2f}s | Saved model: {args.output}")


if __name__ == "__main__":
    main()
