"""Fine-tune DeiT-Tiny on one-channel ESC-50 Log-Mel spectrograms.

Example:
    python 3.py --data-root /path/to/ESC-50-master
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError as error:
    raise ImportError("Install timm first: pip install timm") from error


SEED = 42
SAMPLE_RATE = 44_100
CLIP_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 441
INPUT_SIZE = (224, 224)
NUM_CLASSES = 50


def set_seed(seed: int = SEED) -> None:
    """Make data shuffling and parameter initialization reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ESC50Dataset(Dataset):
    """Load ESC-50 clips and convert each to a normalized Log-Mel image."""

    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
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
        spectrogram = F.interpolate(
            spectrogram.unsqueeze(0), size=INPUT_SIZE, mode="bilinear", align_corners=False
        ).squeeze(0)
        return spectrogram.float(), int(row["target"])


def load_deit_tiny_pretrained() -> nn.Module:
    """Load DeiT-Tiny across common timm model-name versions."""
    errors: list[Exception] = []
    for name in ("deit_tiny_patch16_224.fb_in1k", "deit_tiny_patch16_224"):
        try:
            return timm.create_model(name, pretrained=True)
        except Exception as error:  # Try the compatible fallback model name.
            errors.append(error)
    raise RuntimeError(f"Could not load pretrained DeiT-Tiny: {errors[-1]}")


def build_deit_for_spectrogram(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Adapt pretrained RGB patch projection and head for ESC-50."""
    model = load_deit_tiny_pretrained()
    old_projection = model.patch_embed.proj
    new_projection = nn.Conv2d(
        in_channels=1,
        out_channels=old_projection.out_channels,
        kernel_size=old_projection.kernel_size,
        stride=old_projection.stride,
        padding=old_projection.padding,
        dilation=old_projection.dilation,
        groups=old_projection.groups,
        bias=old_projection.bias is not None,
    )
    with torch.no_grad():
        new_projection.weight.copy_(old_projection.weight.mean(dim=1, keepdim=True))
        if old_projection.bias is not None:
            new_projection.bias.copy_(old_projection.bias)
    model.patch_embed.proj = new_projection
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def configure_deit_finetuning(model: nn.Module, num_unfrozen_blocks: int = 2) -> nn.Module:
    """Freeze DeiT except its last encoder blocks, final norm, and classifier."""
    if not 0 <= num_unfrozen_blocks <= len(model.blocks):
        raise ValueError(f"num_unfrozen_blocks must be in [0, {len(model.blocks)}].")

    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.blocks[-num_unfrozen_blocks:] if num_unfrozen_blocks else []:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += criterion(logits, targets).item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
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
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--unfrozen-blocks", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("deit_tiny_esc50_best.pt"))
    args = parser.parse_args()

    metadata_path = args.data_root / "meta" / "esc50.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_root}. Set --data-root to ESC-50-master.")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 2 if device.type == "cuda" else 0
    print(f"Device: {device}")

    metadata = pd.read_csv(metadata_path)
    train_data = metadata[metadata["fold"].isin((1, 2, 3))]
    validation_data = metadata[metadata["fold"] == 4]
    test_data = metadata[metadata["fold"] == 5]
    print(f"Train: {len(train_data)} | Validation: {len(validation_data)} | Test: {len(test_data)}")

    options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    audio_dir = args.data_root / "audio"
    train_loader = DataLoader(ESC50Dataset(train_data, audio_dir), args.batch_size, shuffle=True, **options)
    validation_loader = DataLoader(ESC50Dataset(validation_data, audio_dir), args.batch_size, shuffle=False, **options)
    test_loader = DataLoader(ESC50Dataset(test_data, audio_dir), args.batch_size, shuffle=False, **options)

    model = configure_deit_finetuning(build_deit_for_spectrogram(), args.unfrozen_blocks).to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    best_validation_accuracy, training_seconds = train(
        model, train_loader, validation_loader, optimizer, criterion, device, args.epochs
    )
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    torch.save(model.state_dict(), args.output)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_seconds:.2f}s | Saved model: {args.output}")


if __name__ == "__main__":
    main()
