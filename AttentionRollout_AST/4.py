"""Visualise AST [CLS] attention on ESC-50 spectrograms.

Example:
    python attention_rollout_ast.py --esc50-root /content/ESC-50-master
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification


AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORIES = {
    "train_whistle": "train",
    "chirping_birds": "chirping_birds",
    "rain": "rain",
}


def load_ast_model_and_extractor(checkpoint: str = AST_CHECKPOINT):
    """Load AST with eager attention, which is required to return weights."""
    model = ASTForAudioClassification.from_pretrained(
        checkpoint,
        attn_implementation="eager",
    )
    feature_extractor = ASTFeatureExtractor.from_pretrained(checkpoint)
    model.config.output_attentions = True
    model = model.to(DEVICE).eval()
    return model, feature_extractor


def find_esc50_root(requested_root: str | None) -> Path:
    """Find the directory containing ``audio/`` and ``meta/esc50.csv``."""
    if requested_root is not None:
        root = Path(requested_root).expanduser().resolve()
        metadata = root / "meta" / "esc50.csv"
        if metadata.exists() and (root / "audio").is_dir():
            return root
        raise FileNotFoundError(f"ESC-50 was not found at: {root}")

    search_roots = (Path.cwd(), Path("/content"))
    for search_root in search_roots:
        if search_root.exists():
            for metadata in search_root.rglob("esc50.csv"):
                root = metadata.parent.parent
                if (root / "audio").is_dir():
                    return root
    raise FileNotFoundError(
        "Cannot find ESC-50. Pass its directory with --esc50-root, "
        "for example: /content/ESC-50-master"
    )


def select_sample_files(metadata_file: Path) -> dict[str, str]:
    """Select one real ESC-50 audio filename for every requested category."""
    wanted = set(CATEGORIES.values())
    filenames: dict[str, str] = {}
    with metadata_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            category = row["category"]
            if category in wanted and category not in filenames:
                filenames[category] = row["filename"]

    missing = wanted - filenames.keys()
    if missing:
        raise ValueError(f"Missing categories in esc50.csv: {sorted(missing)}")
    return {label: filenames[category] for label, category in CATEGORIES.items()}


def load_waveform(path: Path, target_sr: int = AST_SAMPLE_RATE) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.squeeze(0)


def infer_patch_grid(model) -> tuple[int, int]:
    config = model.config
    patch_size = config.patch_size
    frequency_patches = (config.num_mel_bins - patch_size) // config.frequency_stride + 1
    time_patches = (config.max_length - patch_size) // config.time_stride + 1
    return frequency_patches, time_patches


@torch.no_grad()
def get_cls_attention_last_layer(model, feature_extractor, waveform: torch.Tensor):
    """Return [CLS]-to-patch attention and the normalised input spectrogram."""
    inputs = feature_extractor(
        waveform.detach().cpu().numpy(),
        sampling_rate=AST_SAMPLE_RATE,
        return_tensors="pt",
    )
    inputs = {name: value.to(DEVICE) for name, value in inputs.items()}
    outputs = model(**inputs, output_attentions=True)

    # AST has both CLS and distillation tokens. Infer their number instead of
    # hard-coding it, so the patch vector always has the expected length.
    attention = outputs.attentions[-1].mean(dim=1)[0]  # (sequence, sequence)
    frequency_patches, time_patches = infer_patch_grid(model)
    patch_count = frequency_patches * time_patches
    special_token_count = attention.shape[-1] - patch_count
    if special_token_count < 1:
        raise RuntimeError("The AST sequence is shorter than its patch grid.")

    return attention[0, special_token_count:], inputs["input_values"]


def attention_to_spectrogram(attention_vector: torch.Tensor, model, spectrogram_shape):
    """Unpatchify attention and return it in spectrogram order: (time, freq)."""
    frequency_patches, time_patches = infer_patch_grid(model)
    attention_2d = attention_vector.reshape(frequency_patches, time_patches)

    # AST input_values is (batch, time, mel). The patch grid is (freq, time).
    time_bins, frequency_bins = spectrogram_shape
    attention_freq_time = F.interpolate(
        attention_2d[None, None],
        size=(frequency_bins, time_bins),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return attention_freq_time.T


def plot_attention(name: str, waveform: torch.Tensor, model, feature_extractor, output_dir: Path):
    attention_vector, input_values = get_cls_attention_last_layer(
        model, feature_extractor, waveform
    )
    spectrogram = input_values.squeeze(0).detach().cpu()
    attention_map = attention_to_spectrogram(attention_vector, model, spectrogram.shape).cpu()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, image, title, cmap in (
        (axes[0], spectrogram.T, "Spectrogram", "magma"),
        (axes[1], attention_map.T, "CLS attention (last layer)", "viridis"),
    ):
        axis.imshow(image, origin="lower", aspect="auto", cmap=cmap)
        axis.set_title(f"{name} — {title}")
        axis.set_xlabel("Time")
        axis.set_ylabel("Mel frequency")

    axes[2].imshow(spectrogram.T, origin="lower", aspect="auto", cmap="magma")
    axes[2].imshow(attention_map.T, origin="lower", aspect="auto", cmap="viridis", alpha=0.5)
    axes[2].set_title(f"{name} — Overlay")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Mel frequency")
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}_attention.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def attention_rollout(all_layer_attentions: list[torch.Tensor], special_token_count: int = 2):
    """Optional: aggregate mean-head attentions across all transformer layers."""
    result = None
    for attention in all_layer_attentions:
        identity = torch.eye(attention.size(-1), device=attention.device, dtype=attention.dtype)
        augmented_attention = 0.5 * attention + 0.5 * identity
        result = augmented_attention if result is None else augmented_attention @ result
    if result is None:
        raise ValueError("all_layer_attentions must not be empty")
    return result[0, special_token_count:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esc50-root", help="Directory such as /content/ESC-50-master")
    parser.add_argument("--output-dir", default="attention_outputs")
    parser.add_argument("--checkpoint", default=AST_CHECKPOINT)
    args = parser.parse_args()

    esc50_root = find_esc50_root(args.esc50_root)
    audio_dir = esc50_root / "audio"
    selected_files = select_sample_files(esc50_root / "meta" / "esc50.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"ESC-50: {esc50_root}")
    print(f"Samples: {selected_files}")
    model, feature_extractor = load_ast_model_and_extractor(args.checkpoint)

    for name, filename in selected_files.items():
        audio_file = audio_dir / filename
        if not audio_file.is_file():
            raise FileNotFoundError(f"Missing audio file: {audio_file}")
        plot_attention(name, load_waveform(audio_file), model, feature_extractor, output_dir)


if __name__ == "__main__":
    main()
