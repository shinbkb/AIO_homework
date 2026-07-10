"""
=============================================================
  HAAR CASCADE - CODE CHAY TỪ ĐẦU (Educational Version)
=============================================================
Mục tiêu: Hiểu từng bước của thuật toán Haar Cascade
  1. Integral Image
  2. Haar-like Features
  3. Weak Classifier (Decision Stump)
  4. AdaBoost Training
  5. Cascade Classifier
  6. Sliding Window Detection
=============================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import product


# ─────────────────────────────────────────────
# BƯỚC 1: INTEGRAL IMAGE (Ảnh tích phân)
# ─────────────────────────────────────────────
def compute_integral_image(img):
    """
    Tính ảnh tích phân: ii[y,x] = tổng tất cả pixel từ (0,0) đến (y,x)
    Dùng để tính tổng bất kỳ vùng chữ nhật chỉ với 4 phép toán.
    """
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii, r, c, height, width):
    """
    Tính tổng pixel trong hình chữ nhật [r:r+h, c:c+w]
    Sử dụng công thức: A - B - C + D từ ảnh tích phân
    
         c      c+w
    r    A───────B
         │  vùng│
    r+h  C───────D
    
    SUM = D - B - C + A
    """
    r2, c2 = r + height - 1, c + width - 1

    # Clamp để không vượt biên
    r  = max(r, 0);  c  = max(c, 0)
    r2 = min(r2, ii.shape[0] - 1)
    c2 = min(c2, ii.shape[1] - 1)

    D = ii[r2, c2]
    B = ii[r - 1, c2] if r > 0 else 0
    C = ii[r2, c - 1] if c > 0 else 0
    A = ii[r - 1, c - 1] if (r > 0 and c > 0) else 0

    return D - B - C + A


# ─────────────────────────────────────────────
# BƯỚC 2: HAAR-LIKE FEATURES
# ─────────────────────────────────────────────
class HaarFeature:
    """
    Một Haar-like feature bao gồm:
    - type: loại feature ('2h', '2v', '3h', '3v', '4')
    - position: (row, col) góc trên-trái
    - size: (height, width)
    
    Giá trị = Tổng vùng SÁNG - Tổng vùng TỐI
    """
    TYPES = ['2h', '2v', '3h', '3v', '4']

    def __init__(self, feat_type, pos, size):
        self.type = feat_type   # loại feature
        self.pos  = pos         # (row, col)
        self.size = size        # (height, width)

    def compute(self, ii):
        """Tính giá trị feature từ integral image"""
        r, c = self.pos
        h, w = self.size

        if self.type == '2h':
            # Hai cột ngang: trái - phải
            half_w = w // 2
            left  = rect_sum(ii, r, c,          h, half_w)
            right = rect_sum(ii, r, c + half_w, h, half_w)
            return left - right

        elif self.type == '2v':
            # Hai hàng dọc: trên - dưới
            half_h = h // 2
            top    = rect_sum(ii, r,          c, half_h, w)
            bottom = rect_sum(ii, r + half_h, c, half_h, w)
            return top - bottom

        elif self.type == '3h':
            # Ba cột: trái - giữa + phải (giữa là vùng tối)
            third_w = w // 3
            left   = rect_sum(ii, r, c,             h, third_w)
            mid    = rect_sum(ii, r, c + third_w,   h, third_w)
            right  = rect_sum(ii, r, c + 2*third_w, h, third_w)
            return left - mid + right

        elif self.type == '3v':
            # Ba hàng dọc
            third_h = h // 3
            top    = rect_sum(ii, r,             c, third_h, w)
            mid    = rect_sum(ii, r + third_h,   c, third_h, w)
            bottom = rect_sum(ii, r + 2*third_h, c, third_h, w)
            return top - mid + bottom

        elif self.type == '4':
            # Bốn ô checkerboard
            half_h, half_w = h // 2, w // 2
            top_left  = rect_sum(ii, r,          c,          half_h, half_w)
            top_right = rect_sum(ii, r,          c + half_w, half_h, half_w)
            bot_left  = rect_sum(ii, r + half_h, c,          half_h, half_w)
            bot_right = rect_sum(ii, r + half_h, c + half_w, half_h, half_w)
            return (top_left + bot_right) - (top_right + bot_left)

        return 0


def generate_features(win_size=24):
    """
    Tạo tất cả Haar features có thể trong cửa sổ win_size x win_size.
    (Với win_size=24, có ~160,000 features — ở đây ta giới hạn để nhanh hơn)
    """
    features = []
    for feat_type in HaarFeature.TYPES:
        # Kích thước tối thiểu tùy loại feature
        min_w = 4 if 'h' in feat_type or feat_type == '4' else 2
        min_h = 4 if 'v' in feat_type or feat_type == '4' else 2
        if feat_type == '3h': min_w = 6
        if feat_type == '3v': min_h = 6

        for h in range(min_h, win_size + 1, 2):
            for w in range(min_w, win_size + 1, 2):
                for r in range(0, win_size - h + 1, 2):
                    for c in range(0, win_size - w + 1, 2):
                        features.append(HaarFeature(feat_type, (r, c), (h, w)))

    print(f"Tổng số features tạo ra: {len(features):,}")
    return features


# ─────────────────────────────────────────────
# BƯỚC 3: WEAK CLASSIFIER (Decision Stump)
# ─────────────────────────────────────────────
class WeakClassifier:
    """
    Phân loại yếu (weak classifier) dạng Decision Stump:
    - Chọn 1 Haar feature
    - Chọn 1 ngưỡng threshold
    - Phân loại: +1 nếu feature_value * polarity < threshold * polarity
    """
    def __init__(self):
        self.feature   = None
        self.threshold = 0
        self.polarity  = 1   # +1 hoặc -1
        self.alpha     = 0   # trọng số trong AdaBoost

    def predict(self, feat_val):
        """Dự đoán nhãn (+1 mặt, -1 không mặt) từ giá trị feature"""
        if self.polarity * feat_val < self.polarity * self.threshold:
            return 1
        return -1

    def train(self, feat_vals, labels, weights):
        """
        Train trên một feature duy nhất.
        feat_vals: mảng giá trị feature cho mỗi mẫu
        labels:    mảng nhãn (+1 hoặc -1)
        weights:   mảng trọng số mẫu (tổng = 1)
        Trả về weighted error thấp nhất tìm được.
        """
        n = len(labels)
        pos_weights = weights[labels == 1].sum()   # tổng w của mẫu dương
        neg_weights = weights[labels == -1].sum()  # tổng w của mẫu âm

        min_error = float('inf')
        sorted_idx = np.argsort(feat_vals)

        # Tổng tích lũy để tìm threshold hiệu quả
        cum_pos = 0.0
        cum_neg = 0.0

        for i in sorted_idx:
            # Threshold = feat_vals[i]
            # Polarity +1: predict +1 nếu < threshold
            err_p1 = cum_pos + (neg_weights - cum_neg)
            # Polarity -1: predict +1 nếu > threshold
            err_m1 = cum_neg + (pos_weights - cum_pos)

            for polarity, error in [(1, err_p1), (-1, err_m1)]:
                if error < min_error:
                    min_error       = error
                    self.threshold  = feat_vals[i]
                    self.polarity   = polarity

            if labels[i] == 1:
                cum_pos += weights[i]
            else:
                cum_neg += weights[i]

        return min_error


# ─────────────────────────────────────────────
# BƯỚC 4: ADABOOST TRAINING
# ─────────────────────────────────────────────
def adaboost_train(X, y, features, n_weak=10):
    """
    Train AdaBoost để chọn n_weak weak classifiers tốt nhất.

    X: ma trận (n_samples, n_features) — giá trị Haar features
    y: mảng nhãn +1/-1
    features: danh sách HaarFeature tương ứng
    n_weak: số weak classifiers cần chọn

    Trả về danh sách (WeakClassifier, alpha)
    """
    n = len(y)
    weights = np.ones(n) / n   # Khởi tạo trọng số đều nhau
    classifiers = []

    for t in range(n_weak):
        # Chuẩn hóa weights
        weights /= weights.sum()

        best_clf   = WeakClassifier()
        best_error = float('inf')
        best_feat_idx = 0

        # Tìm weak classifier tốt nhất trên tất cả features
        for j in range(X.shape[1]):
            clf = WeakClassifier()
            clf.feature = features[j]
            error = clf.train(X[:, j], y, weights)
            if error < best_error:
                best_error    = error
                best_clf      = clf
                best_clf.feature = features[j]
                best_feat_idx = j
                # Cập nhật threshold và polarity tốt nhất
                best_clf.threshold = clf.threshold
                best_clf.polarity  = clf.polarity

        # Tính alpha (trọng số của weak classifier này)
        eps = max(best_error, 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)
        best_clf.alpha = alpha

        # Cập nhật weights: tăng cho mẫu sai, giảm cho mẫu đúng
        preds = np.array([best_clf.predict(X[i, best_feat_idx]) for i in range(n)])
        weights *= np.exp(-alpha * y * preds)

        classifiers.append(best_clf)
        feat_type = features[best_feat_idx].type
        print(f"  Vòng {t+1:2d}: feature={feat_type}, "
              f"threshold={best_clf.threshold:.1f}, "
              f"alpha={alpha:.3f}, error={best_error:.4f}")

    return classifiers


# ─────────────────────────────────────────────
# BƯỚC 5: STRONG CLASSIFIER (kết hợp weak)
# ─────────────────────────────────────────────
def strong_predict(classifiers, ii, threshold_factor=0.5):
    """
    Kết hợp các weak classifiers với trọng số alpha.
    H(x) = sign( Σ alpha_t * h_t(x) )

    threshold_factor: điều chỉnh ngưỡng quyết định
      - < 0.5 → giảm ngưỡng → tăng recall (ít bỏ sót)
      - > 0.5 → tăng ngưỡng → tăng precision (ít false positive)
    """
    total_alpha = sum(clf.alpha for clf in classifiers)
    score = 0.0

    for clf in classifiers:
        val = clf.feature.compute(ii)
        score += clf.alpha * clf.predict(val)

    # Ngưỡng mặc định = 0 (sign), có thể điều chỉnh
    threshold = (2 * threshold_factor - 1) * total_alpha
    return 1 if score > threshold else -1, score


# ─────────────────────────────────────────────
# BƯỚC 6: SLIDING WINDOW DETECTION
# ─────────────────────────────────────────────
def sliding_window_detect(img_gray, classifiers, win_size=24,
                           scale_factor=1.2, step=4):
    """
    Trượt cửa sổ qua ảnh ở nhiều tỉ lệ để tìm khuôn mặt.
    """
    detections = []
    h_img, w_img = img_gray.shape
    scale = 1.0

    while True:
        # Thu nhỏ ảnh theo tỉ lệ hiện tại
        new_h = int(h_img / scale)
        new_w = int(w_img / scale)
        if new_h < win_size or new_w < win_size:
            break

        resized = cv2.resize(img_gray, (new_w, new_h))

        # Trượt cửa sổ
        for r in range(0, new_h - win_size, step):
            for c in range(0, new_w - win_size, step):
                window = resized[r:r+win_size, c:c+win_size].astype(np.float64)
                ii = compute_integral_image(window)
                label, score = strong_predict(classifiers, ii)

                if label == 1:
                    # Chuyển tọa độ về ảnh gốc
                    x = int(c * scale)
                    y = int(r * scale)
                    s = int(win_size * scale)
                    detections.append((x, y, s, s, score))

        scale *= scale_factor

    return detections


def non_max_suppression(detections, overlap_thresh=0.3):
    """
    Loại bỏ các bounding box trùng lặp (Non-Maximum Suppression).
    Giữ lại box có score cao nhất, loại bỏ box có IoU > overlap_thresh.
    """
    if not detections:
        return []

    boxes = np.array([(x, y, x+w, y+h, score)
                      for x, y, w, h, score in detections], dtype=float)
    x1, y1, x2, y2, scores = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= overlap_thresh]

    return [(int(x1[i]), int(y1[i]),
             int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in keep]


# ─────────────────────────────────────────────
# DEMO: Minh họa từng bước
# ─────────────────────────────────────────────
def demo_integral_image():
    """Minh họa integral image trên ma trận nhỏ"""
    print("\n" + "="*50)
    print("DEMO 1: INTEGRAL IMAGE")
    print("="*50)

    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9,10,11,12],
        [13,14,15,16]
    ], dtype=float)

    ii = compute_integral_image(img)

    print("Ảnh gốc:")
    print(img)
    print("\nẢnh tích phân:")
    print(ii)

    # Tính tổng vùng [1:3, 1:3] (hàng 1-2, cột 1-2)
    s = rect_sum(ii, 1, 1, 2, 2)
    expected = img[1:3, 1:3].sum()
    print(f"\nTổng vùng [1:3, 1:3] bằng integral image: {s}")
    print(f"Tổng vùng [1:3, 1:3] tính trực tiếp:      {expected}")
    assert s == expected, "Sai!"
    print("✓ Kết quả khớp!")


def demo_haar_features():
    """Minh họa Haar features trên ảnh khuôn mặt giả"""
    print("\n" + "="*50)
    print("DEMO 2: HAAR-LIKE FEATURES")
    print("="*50)

    # Tạo ảnh giả: vùng mắt tối hơn má
    win = np.ones((24, 24), dtype=float) * 200   # nền sáng
    win[8:16, 4:20] = 80                          # vùng mắt tối

    ii = compute_integral_image(win)

    # Feature nằm ngang: phát hiện cạnh mắt-má
    feat_2v = HaarFeature('2v', (4, 0), (16, 24))  # trên vs dưới
    val = feat_2v.compute(ii)
    print(f"Feature 2v (đường ngang trên-dưới): {val:.0f}")
    print("  → Dương: vùng trên sáng hơn vùng mắt (đúng với khuôn mặt)")

    feat_2h = HaarFeature('2h', (8, 0), (8, 24))   # trái vs phải
    val2 = feat_2h.compute(ii)
    print(f"Feature 2h (đường dọc trái-phải): {val2:.0f}")
    print("  → ~0: vùng mắt đối xứng trái-phải (đúng)")


def demo_adaboost_simple():
    """Demo AdaBoost với dataset giả đơn giản"""
    print("\n" + "="*50)
    print("DEMO 3: ADABOOST (simplified)")
    print("="*50)

    np.random.seed(42)
    n_pos, n_neg = 20, 20

    # Tạo dữ liệu: mặt người vs không phải mặt
    # Feature 0: giá trị cao → mặt (phân biệt tốt)
    # Feature 1: nhiễu (phân biệt kém)
    X_pos = np.column_stack([
        np.random.normal(100, 20, n_pos),   # feature 0: cao
        np.random.normal(50,  30, n_pos),   # feature 1: ngẫu nhiên
    ])
    X_neg = np.column_stack([
        np.random.normal(-50, 20, n_neg),   # feature 0: thấp
        np.random.normal(45,  30, n_neg),   # feature 1: ngẫu nhiên
    ])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*n_pos + [-1]*n_neg)

    # Tạo các "dummy" feature objects
    class DummyFeature:
        def __init__(self, idx):
            self.type = f"feat_{idx}"
    dummy_features = [DummyFeature(i) for i in range(X.shape[1])]

    # Train 3 weak classifiers
    print("Training AdaBoost với 3 weak classifiers...")
    n = len(y)
    weights = np.ones(n) / n
    classifiers_info = []

    for t in range(3):
        weights /= weights.sum()
        best_error = float('inf')
        best_thresh = 0
        best_pol = 1
        best_feat = 0

        for j in range(X.shape[1]):
            clf = WeakClassifier()
            clf.feature = dummy_features[j]
            error = clf.train(X[:, j], y, weights)
            if error < best_error:
                best_error  = error
                best_thresh = clf.threshold
                best_pol    = clf.polarity
                best_feat   = j

        eps = max(best_error, 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)

        # Dự đoán bằng threshold và polarity tốt nhất vừa tìm được
        feat_vals = X[:, best_feat]
        preds = np.array([1 if best_pol * v < best_pol * best_thresh else -1
                          for v in feat_vals])

        # Cập nhật weights: tăng mẫu sai, giảm mẫu đúng
        weights *= np.exp(-alpha * y * preds)

        classifiers_info.append((best_feat, best_thresh, best_pol, alpha))
        accuracy = (preds == y).mean()
        print(f"Vòng {t+1}: Chọn feature {best_feat}, "
              f"threshold={best_thresh:.1f}, alpha={alpha:.3f}, "
              f"accuracy={accuracy:.0%}")

    # Dự đoán cuối cùng (strong classifier)
    final_score = np.zeros(n)
    for feat_idx, thresh, pol, alpha in classifiers_info:
        preds = np.where(pol * X[:, feat_idx] < pol * thresh, 1, -1)
        final_score += alpha * preds

    final_pred = np.sign(final_score)
    final_acc  = (final_pred == y).mean()
    print(f"\nStrong Classifier accuracy: {final_acc:.0%}")


if __name__ == "__main__":
    # ── Chạy các demo ──
    demo_integral_image()
    demo_haar_features()
    demo_adaboost_simple()

    # ── Visualize Haar features ──
    print("\n" + "="*50)
    print("VISUALIZE: Các loại Haar Features")
    print("="*50)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ['2h (ngang)', '2v (dọc)', '3h (3 cột)', '3v (3 hàng)', '4 (ô bàn cờ)']
    patterns = [
        np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0]], dtype=float),       # 2h
        np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=float),       # 2v
        np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[1,1,0,0,1,1]], dtype=float),    # 3h
        np.array([[1,1,1],[1,1,1],[0,0,0],[0,0,0],[1,1,1],[1,1,1]], dtype=float),# 3v
        np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]], dtype=float),       # 4
    ]

    for ax, title, pat in zip(axes, titles, patterns):
        ax.imshow(pat, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        # Thêm chú thích trắng/đen
        ax.text(0.02, 0.02, '■ = sáng (+)\n□ = tối (-)',
                transform=ax.transAxes, fontsize=7, color='red',
                verticalalignment='bottom')

    plt.suptitle('5 loại Haar-like Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('haar_features_visualization.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("→ Đã lưu: haar_features_visualization.png")

    print("\n✓ Chạy xong tất cả demo!")
    print("\nGhi chú:")
    print("  - Để detect ảnh thực: cần train trên dataset (VD: LFW, FDDB)")
    print("  - Hoặc dùng pipeline self-implement + load weights từ OpenCV XML")
