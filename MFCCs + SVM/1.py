"""Speech Commands classification with MFCC features and an RBF SVM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchaudio
import torchaudio.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from torch.nn.functional import pad


TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SAMPLE_RATE = 16_000
MAX_LENGTH = 16_000
SEED = 42


def filter_dataset(dataset, target_words: list[str]) -> list[tuple[torch.Tensor, int, int]]:
    """Keep target words and convert labels to integer ids."""
    word_to_index = {word: index for index, word in enumerate(target_words)}
    filtered = []
    for waveform, sample_rate, label, *_ in dataset:
        if label in word_to_index:
            filtered.append((waveform, sample_rate, word_to_index[label]))
    return filtered


def build_mfcc_transform(sample_rate: int = SAMPLE_RATE) -> transforms.MFCC:
    return transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
        },
    )


def extract_mfcc_features(
    dataset: list[tuple[torch.Tensor, int, int]],
    mfcc_transform: transforms.MFCC,
    max_length: int = MAX_LENGTH,
    target_sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize waveform lengths and mean-pool MFCCs over time."""
    features = []
    labels = []

    for waveform, sample_rate, label in dataset:
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
            )

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.size(1) < max_length:
            waveform = pad(waveform, (0, max_length - waveform.size(1)))
        else:
            waveform = waveform[:, :max_length]

        mfcc = mfcc_transform(waveform)
        pooled = mfcc.mean(dim=-1).squeeze(0)
        features.append(pooled.numpy())
        labels.append(label)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_words: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(target_words))),
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_words,
        yticklabels=target_words,
    )
    plt.title("Confusion Matrix - MFCC + SVM")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="speech_commands_data")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--output", default="mfcc_svm_confusion_matrix.png")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = Path(args.data_root)
    print("Loading Speech Commands dataset...")
    train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_root,
        download=True,
        subset="training",
    )
    test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=data_root,
        download=False,
        subset="testing",
    )

    train_data = filter_dataset(train_dataset, TARGET_WORDS)
    test_data = filter_dataset(test_dataset, TARGET_WORDS)
    if args.max_train > 0:
        train_data = train_data[: args.max_train]
    if args.max_test > 0:
        test_data = test_data[: args.max_test]

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples : {len(test_data)}")

    mfcc_transform = build_mfcc_transform()
    X_train, y_train = extract_mfcc_features(train_data, mfcc_transform)
    X_test, y_test = extract_mfcc_features(test_data, mfcc_transform)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")

    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        random_state=SEED,
    )
    print("Training SVM...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(TARGET_WORDS))),
            target_names=TARGET_WORDS,
            zero_division=0,
        )
    )

    output_path = Path(args.output)
    plot_confusion_matrix(y_test, predictions, TARGET_WORDS, output_path)
    print(f"Saved confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
