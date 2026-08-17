import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import open_clip
from peft import get_peft_model, LoraConfig

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False


# ==============================================================================
# 1. TIỀN XỬ LÝ & TẢI DỮ LIỆU (KTVIC & UIT-VIC)
# ==============================================================================

def load_coco_dataset(base_path, dataset_name):
    """
    Đọc dữ liệu từ định dạng COCO JSON và liên kết hình ảnh với câu caption tương ứng.
    """
    pairs = []
    json_files = glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True)

    # Cache tất cả đường dẫn ảnh để tra cứu nhanh
    all_image_paths = {}
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                all_image_paths[f] = os.path.join(root, f)

    for json_f in json_files:
        try:
            with open(json_f, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                id_to_filename = {}
                for img in data["images"]:
                    fname = img.get("filename") or img.get("file_name")
                    if fname:
                        id_to_filename[img["id"]] = fname

                for ann in data["annotations"]:
                    img_id = ann.get("image_id")
                    caption = ann.get("caption") or ann.get("segment_caption")
                    if img_id in id_to_filename and caption:
                        fname = id_to_filename[img_id]
                        img_p = all_image_paths.get(fname)
                        if img_p and os.path.exists(img_p):
                            pairs.append({
                                "image_path": img_p,
                                "caption": caption.strip(),
                                "source": dataset_name
                            })
        except Exception as e:
            print(f"[!] Lỗi khi đọc file {json_f}: {e}")
    return pairs


def prepare_data(ktvic_dir=None, uitvic_dir=None):
    """
    Tải hoặc nạp dữ liệu KTVIC & UIT-VIC, sau đó loại bỏ trùng lặp và chia tập dữ liệu.
    """
    print("--- Tải và Nạp Dữ liệu ---")
    if ktvic_dir is None or uitvic_dir is None:
        if not KAGGLEHUB_AVAILABLE:
            raise ImportError("Không tìm thấy `kagglehub`. Vui lòng cài đặt qua `pip install kagglehub` hoặc truyền `--ktvic_dir` và `--uitvic_dir`.")
        print("Đang tải tập dữ liệu từ Kaggle Hub...")
        ktvic_path = ktvic_dir or kagglehub.dataset_download("leo040802/ktvic-dataset")
        uitvic_path = uitvic_dir or kagglehub.dataset_download("leo040802/uitvic-dataset")
    else:
        ktvic_path = ktvic_dir
        uitvic_path = uitvic_dir

    print(f"KTVIC Path: {ktvic_path}")
    print(f"UIT-VIC Path: {uitvic_path}")

    ktvic_pairs = load_coco_dataset(ktvic_path, "KTVIC")
    uitvic_pairs = load_coco_dataset(uitvic_path, "UIT-VIC")

    all_pairs = ktvic_pairs + uitvic_pairs
    print(f"Tổng số cặp (Ảnh - Văn bản) tìm thấy: {len(all_pairs)} (KTVIC: {len(ktvic_pairs)}, UIT-VIC: {len(uitvic_pairs)})")

    df_data = pd.DataFrame(all_pairs)
    df_data = df_data.dropna().drop_duplicates(subset=["caption"]).reset_index(drop=True)
    print(f"Số lượng dữ liệu sau khi loại bỏ trùng lặp: {len(df_data)}")

    # Chia tập dữ liệu 80% Train / 10% Val / 10% Test
    train_df, test_df = train_test_split(df_data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")
    return train_df, val_df, test_df


class ImageTextDataset(Dataset):
    """
    Custom Dataset cho PyTorch chứa cặp Image - Text.
    """
    def __init__(self, df, preprocess, tokenizer):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")
        processed_image = self.preprocess(image)
        tokens = self.tokenizer(caption)[0]

        return {
            "image": processed_image,
            "text": tokens,
            "caption": caption,
            "image_path": image_path
        }


# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH CLIP & LORA PEFT
# ==============================================================================

def build_model(model_name="xlm-roberta-base-ViT-B-32", pretrained_tag="laion5b_s13b_b90k", use_lora=True, device="cuda"):
    """
    Khởi tạo Multilingual CLIP model và tích hợp PEFT LoRA cho Text Encoder.
    """
    print(f"Đang tải mô hình: {model_name} ({pretrained_tag})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    tokenizer = open_clip.get_tokenizer(model_name)

    if use_lora:
        print("Đang áp dụng PEFT LoRA vào Text Encoder...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key", "out_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        model.text.transformer = get_peft_model(model.text.transformer, lora_config)
        print("--- Thông số trainable sau khi áp dụng LoRA ---")
        model.text.transformer.print_trainable_parameters()

    model.to(device)
    return model, preprocess, tokenizer


# ==============================================================================
# 3. HÀM MẤT MÁT (LOSS) VÀ ĐÁNH GIÁ (METRICS)
# ==============================================================================

class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss cho Contrastive Learning giữa Image và Text.
    """
    def __init__(self, logit_scale=100.0):
        super().__init__()
        self.logit_scale = logit_scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        logits_per_image = self.logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T

        labels = torch.arange(len(image_features), device=image_features.device)

        loss_i2t = self.cross_entropy(logits_per_image, labels)
        loss_t2i = self.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2.0


def extract_all_embeddings(model, dataloader, device):
    """
    Trích xuất toàn bộ L2-normalized embeddings cho Image và Text.
    """
    model.eval()
    all_img_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Trích xuất Embeddings", leave=False):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)

            img_feats = model.encode_image(images)
            text_feats = model.encode_text(texts)

            img_feats = F.normalize(img_feats, p=2, dim=-1)
            text_feats = F.normalize(text_feats, p=2, dim=-1)

            all_img_embeds.append(img_feats.cpu())
            all_text_embeds.append(text_feats.cpu())

    return torch.cat(all_img_embeds, dim=0).numpy(), torch.cat(all_text_embeds, dim=0).numpy()


def compute_metrics(image_embeds, text_embeds, k_values=[1, 5, 10]):
    """
    Tính chỉ số Recall@K và MRR cho cả Text-to-Image (T2I) và Image-to-Text (I2T).
    """
    image_embeds = F.normalize(torch.tensor(image_embeds), p=2, dim=-1).numpy()
    text_embeds = F.normalize(torch.tensor(text_embeds), p=2, dim=-1).numpy()

    # Text-to-Image Retrieval
    sim_t2i = text_embeds @ image_embeds.T
    n = sim_t2i.shape[0]

    t2i_ranks = []
    t2i_recalls = {k: 0.0 for k in k_values}
    for i in range(n):
        sorted_idx = np.argsort(-sim_t2i[i])
        rank = np.where(sorted_idx == i)[0][0] + 1
        t2i_ranks.append(rank)
        for k in k_values:
            if rank <= k:
                t2i_recalls[k] += 1.0

    t2i_mrr = np.mean([1.0 / r for r in t2i_ranks])
    metrics = {f"T2I_R@{k}": t2i_recalls[k] / n for k in k_values}
    metrics["T2I_MRR"] = t2i_mrr

    # Image-to-Text Retrieval
    sim_i2t = sim_t2i.T
    i2t_ranks = []
    i2t_recalls = {k: 0.0 for k in k_values}
    for i in range(n):
        sorted_idx = np.argsort(-sim_i2t[i])
        rank = np.where(sorted_idx == i)[0][0] + 1
        i2t_ranks.append(rank)
        for k in k_values:
            if rank <= k:
                i2t_recalls[k] += 1.0

    i2t_mrr = np.mean([1.0 / r for r in i2t_ranks])
    for k in k_values:
        metrics[f"I2T_R@{k}"] = i2t_recalls[k] / n
    metrics["I2T_MRR"] = i2t_mrr

    return metrics


# ==============================================================================
# 4. HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==============================================================================

def train_clip_lora(model, train_loader, val_loader, epochs=5, lr=1e-4, save_dir="checkpoints", device="cuda"):
    """
    Vòng lặp huấn luyện LoRA cho CLIP.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_clip_lora.pt")

    criterion = SymmetricInfoNCELoss(logit_scale=100.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_r1 = 0.0
    history = []

    print("\n--- Bắt đầu Fine-tuning Mô hình CLIP ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch["image"].to(device)
            texts = batch["text"].to(device)

            optimizer.zero_grad()

            img_feats = model.encode_image(images)
            text_feats = model.encode_text(texts)

            loss = criterion(img_feats, text_feats)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Đánh giá trên tập Validation
        val_img_embeds, val_text_embeds = extract_all_embeddings(model, val_loader, device)
        val_metrics = compute_metrics(val_img_embeds, val_text_embeds)
        val_r1 = val_metrics["T2I_R@1"]

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} | Val T2I R@1: {val_r1:.4f} | Val T2I R@5: {val_metrics['T2I_R@5']:.4f} | Val T2I MRR: {val_metrics['T2I_MRR']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            **val_metrics
        })

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            torch.save(model.state_dict(), save_path)
            print(f"--> Đã lưu Best Model Checkpoint tại '{save_path}'!")

    return history, save_path


def visualize_top_k(eval_model, queries, dataset, preprocess, tokenizer, device="cuda", k=5, title_prefix=""):
    """
    Trực quan hóa kết quả tìm kiếm Top-K Text-to-Image.
    """
    eval_model.eval()
    all_images = [Image.open(row["image_path"]).convert("RGB") for _, row in dataset.df.iterrows()]

    img_tensors = torch.stack([preprocess(img) for img in all_images]).to(device)
    with torch.no_grad():
        img_embeddings = eval_model.encode_image(img_tensors)
        img_embeddings = F.normalize(img_embeddings, p=2, dim=-1)

        text_tokens = tokenizer(queries).to(device)
        text_embeddings = eval_model.encode_text(text_tokens)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        sim_matrix = (text_embeddings @ img_embeddings.T).cpu().numpy()

    for idx, query in enumerate(queries):
        scores = sim_matrix[idx]
        top_k_indices = np.argsort(-scores)[:k]

        fig, axes = plt.subplots(1, k, figsize=(18, 4))
        fig.suptitle(f"{title_prefix} Query: '{query}'", fontsize=14, fontweight='bold')

        for rank, img_idx in enumerate(top_k_indices):
            img = all_images[img_idx]
            score = scores[img_idx]

            axes[rank].imshow(img)
            axes[rank].set_title(f"Top-{rank+1} (Score: {score:.3f})")
            axes[rank].axis('off')

        plt.tight_layout()
        plt.show()


# ==============================================================================
# 5. THỰC THI CHÍNH (MAIN FUNCTION)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multilingual CLIP Fine-tuning với PEFT LoRA trên dữ liệu tiếng Việt")
    parser.add_argument("--ktvic_dir", type=str, default=None, help="Đường dẫn thư mục KTVIC dataset (nếu không chọn sẽ tự download qua kagglehub)")
    parser.add_argument("--uitvic_dir", type=str, default=None, help="Đường dẫn thư mục UIT-VIC dataset (nếu không chọn sẽ tự download qua kagglehub)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size khi huấn luyện")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Batch size khi validate/test")
    parser.add_argument("--epochs", type=int, default=5, help="Số lượng epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Thư mục lưu mô hình")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base-ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained_tag", type=str, default="laion5b_s13b_b90k", help="OpenCLIP pretrained tag")
    parser.add_argument("--eval_only", action="store_true", help="Chỉ thực hiện đánh giá baseline và checkpoint có sẵn")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Tải & chuẩn bị dữ liệu
    train_df, val_df, test_df = prepare_data(args.ktvic_dir, args.uitvic_dir)

    # 2. Khởi tạo mô hình
    model, preprocess, tokenizer = build_model(
        model_name=args.model_name,
        pretrained_tag=args.pretrained_tag,
        use_lora=True,
        device=device
    )

    # DataLoaders
    train_dataset = ImageTextDataset(train_df, preprocess, tokenizer)
    val_dataset = ImageTextDataset(val_df, preprocess, tokenizer)
    test_dataset = ImageTextDataset(test_df, preprocess, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    # 3. Đánh giá Zero-shot Baseline
    print("\n--- Đang đánh giá Zero-Shot Baseline trên Test Set ---")
    baseline_model, _, _ = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained_tag)
    baseline_model.to(device)

    zero_img_embeds, zero_text_embeds = extract_all_embeddings(baseline_model, test_loader, device)
    baseline_metrics = compute_metrics(zero_img_embeds, zero_text_embeds)

    print("\n=== KẾT QUẢ ZERO-SHOT BASELINE (TRƯỚC FINE-TUNE) ===")
    for metric, val in baseline_metrics.items():
        print(f"{metric}: {val:.4f}")

    # 4. Huấn luyện Fine-tune
    checkpoint_path = os.path.join(args.save_dir, "best_clip_lora.pt")
    if not args.eval_only:
        history, checkpoint_path = train_clip_lora(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
            device=device
        )

    # 5. Đánh giá sau khi Fine-tune trên Test Set
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\n--- Đã tải checkpoint tốt nhất từ {checkpoint_path} ---")

        ft_img_embeds, ft_text_embeds = extract_all_embeddings(model, test_loader, device)
        finetuned_metrics = compute_metrics(ft_img_embeds, ft_text_embeds)

        comparison_df = pd.DataFrame([
            {"Trạng thái": "Trước Fine-tune (Zero-shot)", **baseline_metrics},
            {"Trạng thái": "Sau Fine-tune (LoRA)", **finetuned_metrics}
        ])

        print("\n=== BẢNG SO SÁNH HIỆU NĂNG TRÊN TẬP TEST ===")
        print(comparison_df.to_string(index=False))
    else:
        print(f"[!] Không tìm thấy checkpoint tại {checkpoint_path}.")

    # 6. Trực quan hóa kết quả tìm kiếm với sample queries
    sample_queries = [
        "Một cô gái mặc áo dài truyền thống đứng bên hồ Hoàn Kiếm",
        "Tách cà phê sữa đá trên bàn gỗ ngoài quán vỉa hè"
    ]
    print("\n--- Trực quan hóa kết quả Top-5 Text-to-Image ---")
    visualize_top_k(model, sample_queries, test_dataset, preprocess, tokenizer, device=device, k=5, title_prefix="[Fine-tuned]")


if __name__ == "__main__":
    main()
