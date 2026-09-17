"""Audio classification with Raw 1D-CNN and Spectrogram 2D-CNN models."""

from __future__ import annotations

import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


SEED = 42
SAMPLE_RATE = 16_000
SAMPLE_LENGTH = SAMPLE_RATE
CLASSES = ["yes", "no", "up", "down"]
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
DATA_ROOT = Path("dataset_speech_commands")
ZIP_PATH = Path("mini_speech_commands.zip")
OUTPUT_FIGURE = Path("audio_classification_results.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class SpeechCommandsDataset(Dataset):
    """Speech Commands subset with mono, 16 kHz, one-second waveforms."""

    def __init__(
        self,
        root_dir: str | Path,
        classes: list[str],
        max_samples_per_class: int = 250,
        target_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_length = target_sample_rate
        self.file_list: list[str] = []
        self.labels: list[int] = []
        label_to_index = {name: index for index, name in enumerate(classes)}

        for class_name in classes:
            files = sorted(
                glob.glob(
                    os.path.join(str(root_dir), "**", class_name, "*.wav"),
                    recursive=True,
                )
            )[:max_samples_per_class]
            self.file_list.extend(files)
            self.labels.extend([label_to_index[class_name]] * len(files))

        if not self.file_list:
            raise FileNotFoundError(f"No .wav files found under {root_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = torchaudio.load(self.file_list[index])

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.target_sample_rate
            )

        if waveform.size(1) < self.target_length:
            waveform = nn.functional.pad(
                waveform, (0, self.target_length - waveform.size(1))
            )
        else:
            waveform = waveform[:, : self.target_length]

        return waveform.float(), self.labels[index]


def download_dataset() -> None:
    """Download and extract Mini Speech Commands when it is unavailable."""
    if not ZIP_PATH.exists() and not DATA_ROOT.exists():
        print("Downloading Mini Speech Commands dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    if not DATA_ROOT.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_ROOT)


def build_transforms() -> tuple[nn.Module, nn.Module, nn.Module]:
    transform_raw = nn.Identity()
    transform_linear = nn.Sequential(
        T.Spectrogram(n_fft=512, hop_length=160),
        T.AmplitudeToDB(),
    )
    transform_logmel = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            n_mels=80,
        ),
        T.AmplitudeToDB(),
    )
    return transform_raw, transform_linear, transform_logmel


class Raw1DCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(start_dim=1)
        return self.classifier(features)


class Audio2DCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(start_dim=1)
        return self.classifier(features)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    transform: nn.Module,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, list]:
    model = model.to(DEVICE)
    transform = transform.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "val_accuracy": []}
    final_targets: list[int] = []
    final_predictions: list[int] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for waveforms, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        ):
            waveforms = waveforms.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(transform(waveforms))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        model.eval()
        correct = 0
        final_targets = []
        final_predictions = []
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(DEVICE)
                labels = labels.to(DEVICE)
                predictions = model(transform(waveforms)).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                final_targets.extend(labels.cpu().tolist())
                final_predictions.extend(predictions.cpu().tolist())

        train_loss = total_loss / total_samples
        val_accuracy = correct / len(val_loader.dataset)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | Val accuracy: {val_accuracy:.4f}"
        )

    history["targets"] = final_targets
    history["predictions"] = final_predictions
    return history


def plot_results(
    histories: dict[str, dict[str, list]],
    sample_wave: torch.Tensor,
    sample_label: int,
    linear_transform: nn.Module,
    logmel_transform: nn.Module,
    output_path: Path,
) -> None:
    linear_spec = linear_transform(sample_wave.unsqueeze(0)).squeeze().cpu().numpy()
    logmel_spec = logmel_transform(sample_wave.unsqueeze(0)).squeeze().cpu().numpy()
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))

    for name, history in histories.items():
        axes[0, 0].plot(
            range(1, len(history["val_accuracy"]) + 1),
            history["val_accuracy"],
            marker="o",
            label=name,
        )
    axes[0, 0].set_title("Validation accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    best_name = "Log-Mel 2D-CNN"
    matrix = confusion_matrix(
        histories[best_name]["targets"],
        histories[best_name]["predictions"],
        labels=list(range(len(CLASSES))),
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Confusion matrix - Log-Mel")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Actual")

    librosa.display.waveshow(
        sample_wave.squeeze().numpy(), sr=SAMPLE_RATE, ax=axes[0, 2]
    )
    axes[0, 2].set_title(f"Waveform - {CLASSES[sample_label]}")

    linear_image = librosa.display.specshow(
        linear_spec,
        sr=SAMPLE_RATE,
        hop_length=160,
        x_axis="time",
        y_axis="linear",
        cmap="magma",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Linear spectrogram (dB)")
    figure.colorbar(linear_image, ax=axes[1, 0], format="%+2.0f dB")

    mel_image = librosa.display.specshow(
        logmel_spec,
        sr=SAMPLE_RATE,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Log-Mel spectrogram (dB)")
    figure.colorbar(mel_image, ax=axes[1, 1], format="%+2.0f dB")

    axes[1, 2].axis("off")
    report = classification_report(
        histories[best_name]["targets"],
        histories[best_name]["predictions"],
        target_names=CLASSES,
        digits=3,
        zero_division=0,
    )
    axes[1, 2].text(0.0, 1.0, report, va="top", family="monospace")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    print(f"Device: {DEVICE}")
    download_dataset()
    dataset = SpeechCommandsDataset(DATA_ROOT, CLASSES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    raw_transform, linear_transform, logmel_transform = build_transforms()

    experiments = {
        "Raw 1D-CNN": (Raw1DCNN(), raw_transform),
        "Linear Spectrogram 2D-CNN": (Audio2DCNN(), linear_transform),
        "Log-Mel 2D-CNN": (Audio2DCNN(), logmel_transform),
    }
    histories = {}
    for name, (model, transform) in experiments.items():
        print(f"\n=== {name} ===")
        histories[name] = train_model(model, train_loader, val_loader, transform)
        print(f"Final validation accuracy: {histories[name]['val_accuracy'][-1]:.4f}")

    print("\n=== Summary ===")
    for name, history in histories.items():
        print(f"{name}: {history['val_accuracy'][-1] * 100:.2f}%")

    sample_wave, sample_label = dataset[0]
    plot_results(
        histories,
        sample_wave,
        sample_label,
        linear_transform,
        logmel_transform,
        OUTPUT_FIGURE,
    )
    print(f"Saved visualization to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
