"""
train_rf.py - Pipeline dự đoán giá vợt cầu lông
================================================
Bước 1: Load & khám phá data
Bước 2: Tiền xử lý (clean, encode)
Bước 3: Train/Test split
Bước 4: Train RandomForestRegressor tự viết
Bước 5: Đánh giá (MAE, RMSE, R²)
Bước 6: Feature Importance chart
Bước 7: So sánh nhanh với sklearn (sanity check)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random_forest import RandomForestRegressor

# ─────────────────────────── CẤU HÌNH ───────────────────────────

CSV_FILE    = "ProjectTabular/data/badminton_hvshop.csv"
TARGET_COL  = "gia_vnd"
FEATURE_COLS = [
    "thuong_hieu",   # brand
    "dong_vot",      # series
    "noi_san_xuat",  # origin
    "diem_can_bang", # balance point
    "do_cung",       # stiffness
    "trong_luong",   # weight
]

# Random Forest hyperparams
N_ESTIMATORS = 100
MAX_DEPTH    = 10
MIN_SAMPLES  = 5
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42
TEST_RATIO   = 0.2

CHART_FILE = "ProjectTabular/data/feature_importance.png"


# ─────────────────────────── HELPER ───────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def train_test_split_manual(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(y)
    idx = rng.permutation(n)
    n_test  = int(n * test_ratio)
    test_i  = idx[:n_test]
    train_i = idx[n_test:]
    return X[train_i], X[test_i], y[train_i], y[test_i]


# ─────────────────────────── 1. LOAD DATA ───────────────────────────

def load_and_preprocess(csv_path: str):
    print("=" * 60)
    print("  PIPELINE DỰ ĐOÁN GIÁ VỢT CẦU LÔNG")
    print("=" * 60)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"\n[1] Đã load {len(df)} hàng từ '{csv_path}'")
    print(f"    Cột: {list(df.columns)}")

    # ── Làm sạch cột giá ──
    # Xóa dấu phẩy/chấm, chuyển sang số
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Loại bỏ hàng không có giá hoặc giá 0
    before = len(df)
    df = df[df[TARGET_COL] > 0].copy()
    print(f"    Bỏ {before - len(df)} hàng thiếu/giá=0 → còn {len(df)} hàng")

    # ── Chọn features ──
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    missing_feats   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        print(f"    ⚠️  Cột không có trong data: {missing_feats}")

    df_feat = df[available_feats].copy()

    # ── Fill missing ──
    for col in df_feat.columns:
        df_feat[col] = df_feat[col].fillna("Unknown")

    # ── Encode categorical → integer codes ──
    encoders = {}  # lưu để dùng lúc predict
    for col in df_feat.columns:
        if df_feat[col].dtype == object:
            codes, uniques = pd.factorize(df_feat[col])
            df_feat[col]   = codes
            encoders[col]  = uniques

    print(f"\n[2] Features sẽ dùng ({len(available_feats)}): {available_feats}")

    X = df_feat.values.astype(float)
    y = df[TARGET_COL].values.astype(float)

    print(f"    Giá: min={y.min():,.0f} | max={y.max():,.0f} | mean={y.mean():,.0f} VNĐ")

    return X, y, available_feats, encoders


# ─────────────────────────── 2. TRAIN & EVAL ───────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, label=""):
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print(f"\n{'─'*40}")
    print(f"  Kết quả: {label}")
    print(f"{'─'*40}")
    print(f"  {'':20} {'Train':>12} {'Test':>12}")
    print(f"  {'MAE (VNĐ)':20} {mae(y_train, pred_train):>12,.0f} {mae(y_test, pred_test):>12,.0f}")
    print(f"  {'RMSE (VNĐ)':20} {rmse(y_train, pred_train):>12,.0f} {rmse(y_test, pred_test):>12,.0f}")
    print(f"  {'R²':20} {r2_score(y_train, pred_train):>12.4f} {r2_score(y_test, pred_test):>12.4f}")

    return mae(y_test, pred_test), r2_score(y_test, pred_test)


# ─────────────────────────── 3. FEATURE IMPORTANCE ───────────────────

def plot_feature_importance(importances, feature_names, save_path):
    order  = np.argsort(importances)
    names  = [feature_names[i] for i in order]
    values = importances[order]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(names, values * 100, color="#4C72B0", edgecolor="white", height=0.6)
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    plt.xlabel("Mức độ ảnh hưởng (%)", fontsize=11)
    plt.title("Feature Importance — Random Forest (tự viết)", fontsize=13, fontweight="bold")
    plt.xlim(0, max(values * 100) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"\n[6] Đã lưu biểu đồ → '{save_path}'")


# ─────────────────────────── 4. SANITY CHECK vs sklearn ──────────────

def sklearn_compare(X_train, X_test, y_train, y_test):
    try:
        from sklearn.ensemble import RandomForestRegressor as SkRF
        sk_model = SkRF(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
        )
        sk_model.fit(X_train, y_train)
        evaluate(sk_model, X_train, X_test, y_train, y_test, "sklearn RF (sanity check)")
    except ImportError:
        print("\n[7] sklearn không có sẵn — bỏ qua sanity check")


# ─────────────────────────── MAIN ───────────────────────────

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Không tìm thấy file: '{CSV_FILE}'")
        print("   Hãy chạy crawl_data.py trước để tạo data.")
        return

    # 1. Tiền xử lý
    X, y, feat_names, encoders = load_and_preprocess(CSV_FILE)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_ratio=TEST_RATIO, seed=RANDOM_STATE
    )
    print(f"\n[3] Split: train={len(y_train)} | test={len(y_test)} (test_ratio={TEST_RATIO})")

    # 3. Train RF tự viết
    print(f"\n[4] Train RandomForest (tự viết): {N_ESTIMATORS} cây, max_depth={MAX_DEPTH}")
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES,
        max_features=MAX_FEATURES,
        bootstrap=True,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    # 4. Đánh giá
    print("\n[5] Đánh giá mô hình:")
    evaluate(rf, X_train, X_test, y_train, y_test, "RandomForest (tự viết)")

    # 5. Feature importance
    print("\nFeature Importances:")
    for name, imp in sorted(zip(feat_names, rf.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:20} {bar:<40} {imp*100:.1f}%")

    os.makedirs(os.path.dirname(CHART_FILE), exist_ok=True)
    plot_feature_importance(rf.feature_importances_, feat_names, CHART_FILE)

    # 6. So sánh với sklearn
    print("\n[7] So sánh với sklearn:")
    sklearn_compare(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("✅ Hoàn thành!")


if __name__ == "__main__":
    main()
