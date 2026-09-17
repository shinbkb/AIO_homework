"""Fine-tune ResNet18 on ESC-50 Log-Mel spectrograms.

Example:
    python train_resnet_esc50.py --download
    python train_resnet_esc50.py --data-root /path/to/ESC-50-master
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
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 441
NUM_CLASSES = 50
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_esc50(data_root: Path) -> None:
    """Download and unpack ESC-50 only when it is not already available."""
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

    if not (data_root / "meta" / "esc50.csv").exists():
        raise RuntimeError("ESC-50 was extracted but its expected files were not found.")


class ESC50Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def _fix_length(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] < NUM_SAMPLES:
            return F.pad(waveform, (0, NUM_SAMPLES - waveform.shape[-1]))
        return waveform[..., :NUM_SAMPLES]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        waveform, sample_rate = torchaudio.load(self.audio_dir / row["filename"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        waveform = self._fix_length(waveform)
        spectrogram = self.db_transform(self.mel_transform(waveform))
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
        return spectrogram.float(), int(row["target"])


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Adapt pretrained ResNet18 from RGB input to a one-channel spectrogram."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def configure_finetuning(model: nn.Module) -> nn.Module:
    """Freeze ResNet except its final block and classification head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, targets).item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()

    sample_count = len(loader.dataset)
    return loss_sum / sample_count, correct / sample_count


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
) -> float:
    best_accuracy = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()

        train_count = len(train_loader.dataset)
        train_loss, train_accuracy = loss_sum / train_count, correct / train_count
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={validation_loss:.4f}, acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("ESC-50-master"))
    parser.add_argument("--download", action="store_true", help="Download ESC-50 if absent.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("resnet18_esc50_best.pt"))
    args = parser.parse_args()

    if args.download:
        download_esc50(args.data_root)
    if not (args.data_root / "meta" / "esc50.csv").exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_root}. Use --download or set --data-root."
        )

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 2 if device.type == "cuda" else 0
    print(f"Device: {device}")

    metadata = pd.read_csv(args.data_root / "meta" / "esc50.csv")
    audio_dir = args.data_root / "audio"
    train_data = metadata[metadata["fold"].isin([1, 2, 3])]
    validation_data = metadata[metadata["fold"] == 4]
    test_data = metadata[metadata["fold"] == 5]
    print(f"Train: {len(train_data)} | Validation: {len(validation_data)} | Test: {len(test_data)}")

    loader_options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(ESC50Dataset(train_data, audio_dir), args.batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(ESC50Dataset(validation_data, audio_dir), args.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(ESC50Dataset(test_data, audio_dir), args.batch_size, shuffle=False, **loader_options)

    model = configure_finetuning(build_model()).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_parameters:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    start = time.perf_counter()
    best_validation_accuracy = train(
        model, train_loader, validation_loader, optimizer, criterion, device, args.epochs
    )
    training_seconds = time.perf_counter() - start
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    torch.save(model.state_dict(), args.output)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_seconds:.2f} s")
    print(f"Saved model: {args.output.resolve()}")


if __name__ == "__main__":
    main()
