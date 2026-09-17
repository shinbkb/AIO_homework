"""GMM-based Voice Activity Detection on Mini Speech Commands."""

from __future__ import annotations

import argparse
import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


SEED = 42
SAMPLE_RATE = 16_000
DATA_ROOT = Path("dataset_speech_commands")
ZIP_PATH = Path("mini_speech_commands.zip")
DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "mini_speech_commands.zip"
)
CLASS_NAMES = ("yes", "no", "up", "down")
BANDS = (
    (80, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
)
OUTPUT_FIGURE = Path("vad_results.png")



def download_dataset() -> None:
    """Download and extract the dataset if it is not available."""
    if not ZIP_PATH.exists() and not DATA_ROOT.exists():
        print("Downloading Mini Speech Commands dataset...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    if not DATA_ROOT.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(DATA_ROOT)



def create_signal(
    data_root: Path,
    seed: int = SEED,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Create a signal containing four speech segments and background noise."""
    rng = np.random.default_rng(seed)
    wav_files: list[str] = []

    for class_name in CLASS_NAMES:
        wav_files.extend(
            glob.glob(
                os.path.join(str(data_root), "**", class_name, "*.wav"),
                recursive=True,
            )
        )

    if len(wav_files) < 4:
        raise FileNotFoundError("At least four speech files are required.")

    selected_files = rng.choice(sorted(wav_files), size=4, replace=False)
    noise_durations = (0.5, 0.4, 0.4, 0.4, 0.4)
    chunks: list[np.ndarray] = []
    speech_intervals: list[tuple[float, float]] = []
    current_sample = 0

    for index, file_path in enumerate(selected_files):
        noise_length = int(sample_rate * noise_durations[index])
        noise = rng.normal(0.0, 0.0008, noise_length).astype(np.float32)
        chunks.append(noise)
        current_sample += noise_length

        speech, _ = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True,
        )
        speech = speech.astype(np.float32)
        start_time = current_sample / sample_rate
        chunks.append(speech)
        current_sample += len(speech)
        end_time = current_sample / sample_rate
        speech_intervals.append((start_time, end_time))

    return np.concatenate(chunks).astype(np.float32), speech_intervals



def extract_subband_features(
    signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_length_ms: int = 20,
    hop_length_ms: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract log-energy features from six frequency bands."""
    frame_length = int(sample_rate * frame_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)

    spectrum = librosa.stft(
        signal,
        n_fft=frame_length,
        hop_length=hop_length,
        win_length=frame_length,
        center=False,
    )
    power_spectrum = np.abs(spectrum) ** 2
    frequencies = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=frame_length,
    )

    features = []
    for low_frequency, high_frequency in BANDS:
        band_mask = (
            (frequencies >= low_frequency)
            & (frequencies < high_frequency)
        )
        band_energy = power_spectrum[band_mask].sum(axis=0)
        features.append(np.log10(band_energy + 1e-10))

    feature_matrix = np.stack(features, axis=1).astype(np.float32)
    frame_times = (
        np.arange(feature_matrix.shape[0]) * hop_length
        + frame_length / 2
    ) / sample_rate
    return feature_matrix, frame_times



def build_ground_truth(
    frame_times: np.ndarray,
    speech_intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Assign one binary ground-truth label to each analysis frame."""
    labels = np.zeros(len(frame_times), dtype=int)
    for index, frame_time in enumerate(frame_times):
        labels[index] = int(
            any(start <= frame_time < end for start, end in speech_intervals)
        )
    return labels



def fit_gmm_vad(
    features: np.ndarray,
    frame_times: np.ndarray,
    threshold: float = 0.0,
    seed: int = SEED,
) -> tuple[GaussianMixture, GaussianMixture, np.ndarray, np.ndarray]:
    """Fit noise/speech GMMs and classify frames using an LLR threshold."""
    noise_features = features[frame_times < 0.4]
    total_energy = features.sum(axis=1)
    speech_cutoff = np.percentile(total_energy, 60)
    speech_features = features[total_energy > speech_cutoff]

    if len(noise_features) < 2 or len(speech_features) < 3:
        raise ValueError("Not enough frames to fit the GMM models.")

    noise_model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=seed,
    ).fit(noise_features)
    speech_model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=seed,
    ).fit(speech_features)

    log_probability_noise = noise_model.score_samples(features)
    log_probability_speech = speech_model.score_samples(features)
    llr = log_probability_speech - log_probability_noise
    predictions = (llr > threshold).astype(int)
    return noise_model, speech_model, llr, predictions



def plot_results(
    signal: np.ndarray,
    sample_rate: int,
    speech_intervals: list[tuple[float, float]],
    frame_times: np.ndarray,
    llr: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
) -> None:
    """Save waveform, VAD, confusion matrix and ROC plots."""
    matrix = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    false_positive_rate, true_positive_rate, _ = roc_curve(
        ground_truth,
        llr,
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)
    time_axis = np.arange(len(signal)) / sample_rate

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(time_axis, signal, color="steelblue", linewidth=0.8)
    for start_time, end_time in speech_intervals:
        axes[0, 0].axvspan(
            start_time,
            end_time,
            color="green",
            alpha=0.25,
        )
    axes[0, 0].set_title("Waveform and ground truth")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(frame_times, llr, color="darkorange")
    axes[0, 1].axhline(0.0, color="black", linestyle="--")
    vad_axis = axes[0, 1].twinx()
    vad_axis.step(
        frame_times,
        predictions,
        where="mid",
        color="red",
        alpha=0.6,
    )
    axes[0, 1].set_title("LLR and GMM VAD")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("LLR")
    vad_axis.set_ylabel("VAD")
    axes[0, 1].grid(alpha=0.3)

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Noise", "Speech"],
        yticklabels=["Noise", "Speech"],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Confusion matrix")
    axes[1, 0].set_xlabel("Predicted label")
    axes[1, 0].set_ylabel("True label")

    axes[1, 1].plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    axes[1, 1].plot(
        [0, 1],
        [0, 1],
        color="navy",
        linestyle="--",
        label="Random classifier",
    )
    axes[1, 1].set_title("ROC curve")
    axes[1, 1].set_xlabel("False positive rate")
    axes[1, 1].set_ylabel("True positive rate")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="LLR threshold used for Speech/Noise classification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FIGURE,
        help="Output image path.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    download_dataset()
    signal, speech_intervals = create_signal(DATA_ROOT)
    features, frame_times = extract_subband_features(signal)
    ground_truth = build_ground_truth(frame_times, speech_intervals)
    _, _, llr, predictions = fit_gmm_vad(
        features,
        frame_times,
        threshold=args.threshold,
    )

    print("=== Classification report ===")
    print(
        classification_report(
            ground_truth,
            predictions,
            target_names=["Noise (0)", "Speech (1)"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(ground_truth, predictions, labels=[0, 1])
    false_positive_rate, true_positive_rate, _ = roc_curve(
        ground_truth,
        llr,
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Confusion matrix:")
    print(matrix)
    print(f"ROC-AUC: {roc_auc:.4f}")

    plot_results(
        signal,
        SAMPLE_RATE,
        speech_intervals,
        frame_times,
        llr,
        predictions,
        ground_truth,
        args.output,
    )
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
