"""
Random Forest Regressor - Cài đặt từ đầu (không dùng sklearn)
=============================================================
Gồm 2 class:
  - DecisionTreeRegressor : cây hồi quy đơn lẻ
  - RandomForestRegressor : tập hợp nhiều cây (ensemble)
"""

import numpy as np


# ─────────────────────────── NODE ───────────────────────────

class DecisionNode:
    """Một node trong cây quyết định."""

    def __init__(
        self,
        feature_idx=None,   # index của feature dùng để split
        threshold=None,     # ngưỡng split
        left=None,          # con trái  (X[feature] <= threshold)
        right=None,         # con phải  (X[feature] >  threshold)
        value=None,         # giá trị dự đoán (chỉ có ở node lá)
        impurity_decrease=0.0,  # mức giảm MSE tại node này
        n_samples=0,        # số mẫu đi qua node này
    ):
        self.feature_idx       = feature_idx
        self.threshold         = threshold
        self.left              = left
        self.right             = right
        self.value             = value
        self.impurity_decrease = impurity_decrease
        self.n_samples         = n_samples

    def is_leaf(self):
        return self.value is not None


# ─────────────────────────── DECISION TREE ───────────────────────────

class DecisionTreeRegressor:
    """
    Cây hồi quy đơn lẻ.

    Tham số
    -------
    max_depth        : độ sâu tối đa (None = không giới hạn)
    min_samples_split: số mẫu tối thiểu để tiếp tục split
    max_features     : số features xét mỗi lần split
                       int  → đúng số đó
                       "sqrt" → sqrt(n_features)
                       "all" / None → tất cả
    random_state     : seed ngẫu nhiên
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state
        self.root              = None
        self._rng              = np.random.default_rng(random_state)
        # Tích lũy impurity decrease theo từng feature (dùng cho feature_importances_)
        self._feat_imp         = None

    # ── Public API ────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Xây cây từ dữ liệu X (n_samples, n_features), y (n_samples,)."""
        n_features = X.shape[1]
        self._feat_imp = np.zeros(n_features)
        self.root = self._build(X, y, depth=0)
        # Chuẩn hóa feature importances về [0, 1]
        total = self._feat_imp.sum()
        if total > 0:
            self._feat_imp /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán từng hàng của X."""
        return np.array([self._predict_row(row, self.root) for row in X])

    @property
    def feature_importances_(self):
        return self._feat_imp

    # ── Private: xây cây ────────────────────────────────────

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape

        # Điều kiện dừng
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.std(y) == 0          # tất cả y như nhau → lá
        ):
            return DecisionNode(value=float(np.mean(y)), n_samples=n_samples)

        # Chọn tập features để xét
        feat_indices = self._sample_features(n_features)

        # Tìm split tốt nhất
        best = self._best_split(X, y, feat_indices)

        if best is None:               # không tách được → lá
            return DecisionNode(value=float(np.mean(y)), n_samples=n_samples)

        feat_idx, threshold, left_mask, impurity_decrease = best

        # Ghi nhận impurity decrease của feature này
        self._feat_imp[feat_idx] += impurity_decrease * n_samples

        # Đệ quy xây con trái/phải
        left_node  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        right_node = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return DecisionNode(
            feature_idx=feat_idx,
            threshold=threshold,
            left=left_node,
            right=right_node,
            impurity_decrease=impurity_decrease,
            n_samples=n_samples,
        )

    def _best_split(self, X, y, feat_indices):
        """
        Duyệt qua các features và ngưỡng để tìm split giảm MSE nhiều nhất.
        Trả về (feature_idx, threshold, left_mask, impurity_decrease) hoặc None.
        """
        parent_mse  = self._mse(y)
        n           = len(y)
        best_gain   = 0.0
        best_split  = None

        for fi in feat_indices:
            col         = X[:, fi]
            thresholds  = np.unique(col)
            if len(thresholds) == 1:
                continue

            # Không thử tất cả ngưỡng — chỉ lấy midpoints cho nhanh
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2

            for thr in midpoints:
                mask_left  = col <= thr
                n_l, n_r   = mask_left.sum(), (~mask_left).sum()
                if n_l == 0 or n_r == 0:
                    continue

                mse_l = self._mse(y[mask_left])
                mse_r = self._mse(y[~mask_left])
                gain  = parent_mse - (n_l / n * mse_l + n_r / n * mse_r)

                if gain > best_gain:
                    best_gain  = gain
                    best_split = (fi, thr, mask_left, gain)

        return best_split  # None nếu không có split nào tốt hơn

    @staticmethod
    def _mse(y):
        """MSE = variance (dùng làm impurity measure)."""
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - np.mean(y)) ** 2))

    def _sample_features(self, n_features):
        """Chọn ngẫu nhiên một subset features."""
        if self.max_features is None or self.max_features == "all":
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = min(int(self.max_features), n_features)
        return self._rng.choice(n_features, size=k, replace=False)

    # ── Private: predict ─────────────────────────────────────

    def _predict_row(self, row, node):
        """Đi từ root xuống lá theo row."""
        if node.is_leaf():
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)


# ─────────────────────────── RANDOM FOREST ───────────────────────────

class RandomForestRegressor:
    """
    Random Forest Regressor từ đầu.

    Tham số
    -------
    n_estimators     : số cây (mặc định 100)
    max_depth        : độ sâu tối đa mỗi cây
    min_samples_split: số mẫu tối thiểu để split
    max_features     : số features random mỗi split ("sqrt" mặc định)
    bootstrap        : True → lấy mẫu bootstrap, False → dùng toàn bộ
    random_state     : seed
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.bootstrap         = bootstrap
        self.random_state      = random_state
        self.trees_            = []          # danh sách (tree, oob_indices)
        self._rng              = np.random.default_rng(random_state)
        self.feature_importances_ = None

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train n_estimators cây, mỗi cây trên 1 bootstrap sample.
        """
        n_samples, n_features = X.shape
        self.trees_ = []
        accum_imp   = np.zeros(n_features)

        for i in range(self.n_estimators):
            seed = int(self._rng.integers(0, 2**31))

            # Bootstrap sampling
            if self.bootstrap:
                idx = self._rng.choice(n_samples, size=n_samples, replace=True)
            else:
                idx = np.arange(n_samples)

            X_boot, y_boot = X[idx], y[idx]

            # Xây cây
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            accum_imp += tree.feature_importances_

            # Log tiến độ mỗi 10 cây
            if (i + 1) % 10 == 0 or (i + 1) == self.n_estimators:
                print(f"  [RF] Đã train {i+1}/{self.n_estimators} cây", end="\r")

        print()  # newline sau progress
        # Feature importances = trung bình của tất cả cây
        self.feature_importances_ = accum_imp / self.n_estimators
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    # ── Predict ──────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Trung bình dự đoán của tất cả cây."""
        all_preds = np.stack([tree.predict(X) for tree in self.trees_], axis=0)
        return all_preds.mean(axis=0)
