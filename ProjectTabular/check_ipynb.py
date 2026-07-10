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


#Chia tap X,y
X = df.drop(columns=['Price']).values.astype(float)
y = np.log1p(df['Price'].values.astype(float))    #log transform price

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


#Train/test split 80/20
np.random.seed(42)      #co dinh bo sinh so ngau nhien

indices = np.random.permutation(len(X))     #chi so sau khi xao tron
split = int(0.8 * len(X))

X_train, X_test = X[indices[:split]],X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
# Import Random Forest từ scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import numpy as np

# 1. Khởi tạo mô hình Random Forest Regressor với cấu hình tương đồng XGBoost
# (Bạn có thể tinh chỉnh các Hyperparameters sao cho phù hợp)
rf_model = RandomForestRegressor(
    n_estimators=500,        # Tổng số cây quyết định (tương tự xgboost)
    max_depth=15,            # Độ sâu tối đa mỗi cây (thường RF để sâu hơn XGBOOST một tý)
    min_samples_split=5,     # Số lượng mẫu tối thiểu để split (tương tự min_child_weight)
    min_samples_leaf=2,      # Số lượng mẫu tối thiểu nằm ở node lá
    max_features='sqrt',     # Lấy ngẫu nhiên căn bậc 2 số features cho mỗi lần split
    random_state=42,         # Cố định random seed
    n_jobs=-1                # Tận dụng tất cả số nhân CPU có trên máy
)

print("Random Forest Model Parameters:")
print(rf_model.get_params())

# 2. Training (Huấn luyện Mô hình)
print("\n" + "="*60)
print("TRAINING RANDOM FOREST MODEL...")
print("="*60)

start_time = time.time()
# Chú ý: Random Forest mặc định không chạy Early Stopping hay truyền eval_set vào hàm fit
# như XGBoost đâu nha, nên code fit() chỉ cần gọi một dòng siêu ngắn thế này thôi
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"\n✓ Training completed!")
print(f"Training time: {training_time:.4f} seconds ({training_time/60:.2f} minutes)")

# 3. Dự đoán (Inference / Prediction)
start_time = time.time()
y_train_pred_rf = rf_model.predict(X_train)
train_pred_time_rf = time.time() - start_time

start_time = time.time()
y_test_pred_rf = rf_model.predict(X_test)
test_pred_time_rf = time.time() - start_time


# 4. Tính toán Metrics Đánh Giá

train_mae_rf = mean_absolute_error(np.expm1(y_train), np.expm1(y_train_pred_rf))
train_mse_rf = mean_squared_error(np.expm1(y_train), np.expm1(y_train_pred_rf))
train_rmse_rf = np.sqrt(train_mse_rf)
train_r2_rf = r2_score(np.expm1(y_train), np.expm1(y_train_pred_rf))

test_mae_rf = mean_absolute_error(np.expm1(y_test), np.expm1(y_test_pred_rf))
test_mse_rf = mean_squared_error(np.expm1(y_test), np.expm1(y_test_pred_rf))
test_rmse_rf = np.sqrt(test_mse_rf)
test_r2_rf = r2_score(np.expm1(y_test), np.expm1(y_test_pred_rf))



# 5. In kết quả (Định dạng giống hệt file Notebook XGBoost nha)
print("\n" + "="*60)
print("RANDOM FOREST MODEL EVALUATION RESULTS")
print("="*60)
print("\n📊 TRAINING SET:")
print(f"  MAE:  {train_mae_rf:,.0f} VND")
print(f"  MSE:  {train_mse_rf:,.0f}")
print(f"  RMSE: {train_rmse_rf:,.0f} VND")
print(f"  R² Score: {train_r2_rf:.4f}")
print(f"  Prediction time: {train_pred_time_rf:.4f} seconds")

print("\n📊 TESTING SET:")
print(f"  MAE:  {test_mae_rf:,.0f} VND")
print(f"  MSE:  {test_mse_rf:,.0f}")
print(f"  RMSE: {test_rmse_rf:,.0f} VND")
print(f"  R² Score: {test_r2_rf:.4f}")
print(f"  Prediction time: {test_pred_time_rf:.4f} seconds")

print("\n📈 OVERALL PERFORMANCE:")
print(f"  Accuracy (R²): {test_r2_rf*100:.2f}%")
print(f"  Average error: ±{test_mae_rf:,.0f} VND")
print(f"  Total time: Training={training_time:.2f}s + Prediction={test_pred_time_rf:.4f}s")

import matplotlib.pyplot as plt

# 1. Giải nén giá tiền thực tế về đơn vị VNĐ để vẽ cho dễ nhìn
y_train_vnd = np.expm1(y_train)
y_train_pred_vnd = np.expm1(y_train_pred_rf)

y_test_vnd = np.expm1(y_test)
y_test_pred_vnd = np.expm1(y_test_pred_rf)

# 2. Bắt đầu vẽ Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Đồ thị 1: Tập Training 
axes[0].scatter(y_train_vnd, y_train_pred_vnd, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
axes[0].plot([y_train_vnd.min(), y_train_vnd.max()], [y_train_vnd.min(), y_train_vnd.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price (VND)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Price (VND)', fontsize=12, fontweight='bold')
axes[0].set_title(f'Random Forest Training Set\nR² = {train_r2_rf:.4f}', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Đồ thị 2: Tập Testing
axes[1].scatter(y_test_vnd, y_test_pred_vnd, alpha=0.5, s=30, edgecolors='k', linewidth=0.5, color='orange')
axes[1].plot([y_test_vnd.min(), y_test_vnd.max()], [y_test_vnd.min(), y_test_vnd.max()], 'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price (VND)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Price (VND)', fontsize=12, fontweight='bold')
axes[1].set_title(f'Random Forest Testing Set\nR² = {test_r2_rf:.4f}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

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
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, n_features='sqrt'):
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