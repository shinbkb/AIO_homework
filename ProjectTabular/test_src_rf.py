import pandas as pd
import numpy as np 
# Bước 1: Load file data mới nhất (v3)
df = pd.read_csv('data/dataprocessing_v3.csv')
df.head()

# Bước 2: Preprocessing dữ liệu v3
print(f"Data trước xử lý: {df.shape}")

# 1. Điền khuyết các cột Tech
df['Tech_Frame'] = df['Tech_Frame'].fillna('None')
df['Tech_Material'] = df['Tech_Material'].fillna('None')
df['Tech_Stability'] = df['Tech_Stability'].fillna('None')

# 2. Xóa các bản ghi NaN trong Price hoặc Max Tension (nếu có để làm sạch)
df = df.dropna().reset_index(drop=True)

# 3. Label Encoding cho cột Version (v1=1, v2=2, v3=3)
version_mapping = {
    'v1': 1, 'v2': 2, 'v3': 3, 'v4': 4, 'v5': 5, 
    'v6': 6, 'v7': 7, 'v8': 8, 'v9': 9, 'v10': 10
}
df['Version'] = df['Version'].map(version_mapping).fillna(1) # Nếu gặp version lạ thì cho là 1 mặc định

# 4. Hàm One-Hot Encoding tự viết
def one_hot_encode(df_target, column):
    unique_vals = sorted(df_target[column].dropna().unique())
    one_hot_df = pd.DataFrame(
        {f"{column}_{val}": (df_target[column] == val).astype(int) for val in unique_vals},
        index=df_target.index
    )
    return pd.concat([df_target.drop(columns=[column]), one_hot_df], axis=1)

# One-Hot Brand, Origin
df = one_hot_encode(df, column='Brand')
df = one_hot_encode(df, column='Origin')

# 5. Hàm Multi-Hot Encoding cho các cột chứa tag list (A, B, C)
def multi_hot_encode(df_target, column):
    all_tags = set()
    for item in df_target[column].dropna():
        tags = [tag.strip() for tag in str(item).split(',')]
        all_tags.update(tags)
        
    if 'None' in all_tags:
        all_tags.remove('None')
    all_tags = sorted(list(all_tags))
    
    multi_hot_dict = {}
    for tag in all_tags:
        col_name = f"{column}_{tag}"
        multi_hot_dict[col_name] = df_target[column].astype(str).apply(
            lambda x: 1 if tag in [t.strip() for t in x.split(',')] else 0
        )
        
    multi_hot_df = pd.DataFrame(multi_hot_dict, index=df_target.index)
    return pd.concat([df_target.drop(columns=[column]), multi_hot_df], axis=1)

# Multi-Hot cho các cột Tech
df = multi_hot_encode(df, column='Tech_Frame')
df = multi_hot_encode(df, column='Tech_Material')
df = multi_hot_encode(df, column='Tech_Stability')

# 6. Target Encoding cho côt Series vì số lượng giá trị duy nhất (unique vals) cao
series_mean_map = df.groupby('Series')['Price'].mean()
df['Series'] = df['Series'].map(series_mean_map)

# Xóa các cột định danh dạng Text còn sót lại mà mô hình RF không hỗ trợ (nếu có)
object_cols = df.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"Bỏ qua các cột text cuối: {object_cols.tolist()}")
    df = df.drop(columns=object_cols)

# In ra các cột cuối cùng của DataFrame sau khi qua chế biến
print("=> Kích thước dataset hiện tại để Train:", df.shape)
df.head()


#Train/test split 80/20
np.random.seed(42)      #co dinh bo sinh so ngau nhien

indices = np.random.permutation(len(X))     #chi so sau khi xao tron
split = int(0.8 * len(X))

X_train, X_test = X[indices[:split]],X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
class DecisionNode:
    def __init__(self,feature=None, threshold = None, left = None, right=None,value = None):
        self.feature = feature
        self.threshold = threshold  #nguong split
        self.left = left    #node trai (<= threshold)
        self.right = right #node phai (> threshold)
        self.value = value  #gia tri du doan (leaf)

class DecisionTree:
    def __init__ (self, max_depth = 10, min_samples_split=5, n_features = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features #so luong feature ngau nhien moi split
        self.root = None
    def mse(self,y):
        return np.mean((y-np.mean(y))**2) if len(y) >0 else 0
    def best_split(self, X, y):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        n_samples, n_cols = X.shape
        # Chọn ngẫu nhiên n_features cột (dùng cho RF)
        feat_ids = np.random.choice(n_cols, self.n_features or n_cols, replace=False)
        parent_mse = self.mse(y)
        for feat in feat_ids:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                # Information gain = giảm MSE sau split
                gain = parent_mse - (
                    left_mask.sum()  / n_samples * self.mse(y[left_mask]) +
                    right_mask.sum() / n_samples * self.mse(y[right_mask])
                )
                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = thresh
        return best_feat, best_thresh
    def build(self, X, y, depth=0):
        # Điều kiện dừng → tạo leaf node
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return DecisionNode(value=np.mean(y))
        feat, thresh = self.best_split(X, y)
        if feat is None:                          # không tìm được split
            return DecisionNode(value=np.mean(y))
        left_mask = X[:, feat] <= thresh
        left  = self.build(X[left_mask],  y[left_mask],  depth + 1)
        right = self.build(X[~left_mask], y[~left_mask], depth + 1)
        return DecisionNode(feature=feat, threshold=thresh, left=left, right=right)
    def fit(self, X, y):
        self.root = self.build(X, y)
    # ── Predict 1 mẫu ──
    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)
    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])
class RandomForestRegressor:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=5, n_features='sqrt'):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_mode  = n_features
        self.trees            = []
    def fit(self, X, y):
        self.trees = []
        n_samples, n_cols = X.shape
        # Số feature mỗi split
        if self.n_features_mode == 'sqrt':
            n_feat = int(np.sqrt(n_cols))
        elif self.n_features_mode == 'log2':
            n_feat = int(np.log2(n_cols))
        else:
            n_feat = n_cols
        for i in range(self.n_estimators):
            # Bootstrap sampling
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_feat
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            if (i + 1) % 10 == 0:
                print(f"  Tree {i+1}/{self.n_estimators} done")
    def predict(self, X):
        # Trung bình prediction của tất cả cây
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=3)
rf.fit(X_train, y_train)
# Predict
y_pred_actual = np.expm1(rf.predict(X_test))
y_test_actual = np.expm1(y_test)# Metrics thủ công
rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual) ** 2))
ss_res = np.sum((y_test_actual - y_pred_actual) ** 2)
ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nRMSE : {rmse:,.0f} VNĐ")
print(f"R²   : {r2:.4f}")