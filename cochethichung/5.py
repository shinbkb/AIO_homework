"""Thích ứng positional embedding 2D từ ViT/DeiT cho AST.

Pipeline:
1. Cắt vùng trung tâm theo trục tần số: 24 -> 12.
2. Nội suy bilinear theo trục thời gian: 24 -> 100.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SEED = 42
SRC_FREQ, SRC_TIME, EMBED_DIM = 24, 24, 768
TGT_FREQ, TGT_TIME = 12, 100


def make_synthetic_pos_embed(
    freq: int = SRC_FREQ,
    time: int = SRC_TIME,
    dim: int = EMBED_DIM,
    seed: int = SEED,
) -> torch.Tensor:
    """Tạo positional embedding giả lập, biến thiên mượt theo không gian."""
    generator = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, freq),
        torch.linspace(0, 2 * np.pi, time),
        indexing="ij",
    )

    freqs_y = torch.rand(dim, generator=generator) * 2 + 0.5
    freqs_x = torch.rand(dim, generator=generator) * 2 + 0.5
    phases = torch.rand(dim, generator=generator) * 2 * np.pi

    pos_embed = torch.empty(freq, time, dim)
    for channel in range(dim):
        pos_embed[:, :, channel] = torch.sin(
            freqs_y[channel] * yy + freqs_x[channel] * xx + phases[channel]
        )

    noise = torch.randn(freq, time, dim, generator=generator) * 0.05
    return pos_embed + noise


def crop_frequency_axis(
    pos_embed: torch.Tensor, target_freq: int = TGT_FREQ
) -> torch.Tensor:
    """Cắt vùng trung tâm của trục tần số.

    Args:
        pos_embed: Tensor có shape (freq, time, dim).
        target_freq: Số patch tần số cần giữ lại.
    """
    freq = pos_embed.shape[0]
    if not 0 < target_freq <= freq:
        raise ValueError("target_freq phải nằm trong khoảng từ 1 đến freq.")

    start = (freq - target_freq) // 2
    return pos_embed[start : start + target_freq, :, :]


def interpolate_time_axis(
    pos_embed: torch.Tensor, target_time: int = TGT_TIME
) -> torch.Tensor:
    """Nội suy bilinear trục thời gian của positional embedding."""
    freq, _, _ = pos_embed.shape
    if target_time <= 0:
        raise ValueError("target_time phải lớn hơn 0.")

    # F.interpolate nhận input theo định dạng (batch, channel, height, width).
    embedding_as_image = pos_embed.permute(2, 0, 1).unsqueeze(0)
    interpolated = F.interpolate(
        embedding_as_image,
        size=(freq, target_time),
        mode="bilinear",
        align_corners=False,
    )
    return interpolated.squeeze(0).permute(1, 2, 0)


def adapt_positional_embedding(
    pos_embed: torch.Tensor,
    target_freq: int = TGT_FREQ,
    target_time: int = TGT_TIME,
) -> torch.Tensor:
    """Cắt tần số và nội suy thời gian cho phù hợp với lưới AST."""
    freq_cropped = crop_frequency_axis(pos_embed, target_freq)
    return interpolate_time_axis(freq_cropped, target_time)


def positional_norm_map(pos_embed: torch.Tensor) -> torch.Tensor:
    """Tính L2 norm của embedding ở mỗi vị trí (freq, time)."""
    return torch.linalg.norm(pos_embed, dim=-1)


def plot_line_profile(
    pos_embed_before: torch.Tensor,
    pos_embed_after: torch.Tensor,
    freq_row: int | None = None,
) -> None:
    """So sánh L2 norm trước và sau nội suy tại một hàng tần số."""
    if freq_row is None:
        freq_row = pos_embed_before.shape[0] // 2
    if not 0 <= freq_row < pos_embed_before.shape[0]:
        raise IndexError("freq_row nằm ngoài phạm vi trục tần số.")

    norm_before = positional_norm_map(pos_embed_before)[freq_row]
    norm_after = positional_norm_map(pos_embed_after)[freq_row]
    x_before = np.linspace(0, 1, norm_before.numel())
    x_after = np.linspace(0, 1, norm_after.numel())

    plt.figure(figsize=(10, 4))
    plt.plot(x_before, norm_before.numpy(), "o-", label="Trước nội suy (24)")
    plt.plot(x_after, norm_after.numpy(), "-", label="Sau nội suy (100)")
    plt.title(f"Line profile tại hàng tần số {freq_row}")
    plt.xlabel("Vị trí thời gian (chuẩn hoá)")
    plt.ylabel("L2 norm")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    torch.manual_seed(SEED)
    pos_embed_src = make_synthetic_pos_embed()
    pos_embed_ast = adapt_positional_embedding(pos_embed_src)

    print("Embedding gốc:", tuple(pos_embed_src.shape))
    print("Embedding AST:", tuple(pos_embed_ast.shape))
    assert pos_embed_ast.shape == (TGT_FREQ, TGT_TIME, EMBED_DIM)

    plot_line_profile(
        crop_frequency_axis(pos_embed_src),
        pos_embed_ast,
        freq_row=6,
    )


if __name__ == "__main__":
    main()
