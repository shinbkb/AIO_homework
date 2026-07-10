import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đã cắt ngưỡng (thresholded image)
# Giả sử bạn đã có ảnh threshold từ code trước
img = cv2.imread('doraemon.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng threshold để có ảnh nhị phân
_, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Chuẩn hóa ảnh về khoảng [0, 1]
normalized = thresholded / 255.0

# Tạo các ảnh với biến đổi hàm mũ với các giá trị μ (gamma) khác nhau
gamma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
transformed_images = []

for gamma in gamma_values:
    # Áp dụng công thức: s = r^μ
    transformed = np.power(normalized, gamma)
    # Chuyển về khoảng [0, 255]
    transformed = np.uint8(transformed * 255)
    transformed_images.append(transformed)

# Hiển thị kết quả
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Biến đổi hàm mũ với các giá trị μ khác nhau', fontsize=16)

# Hiển thị ảnh gốc đã threshold
axes[0, 0].imshow(thresholded, cmap='gray')
axes[0, 0].set_title('Ảnh gốc đã cắt ngưỡng')
axes[0, 0].axis('off')

# Hiển thị các ảnh đã biến đổi
for idx, (gamma, transformed) in enumerate(zip(gamma_values, transformed_images)):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(transformed, cmap='gray')
    axes[row, col].set_title(f'μ = {gamma}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# In giải thích
print("Giải thích:")
print("- μ < 1: Làm sáng ảnh (tăng cường các vùng tối)")
print("- μ = 1: Không thay đổi")
print("- μ > 1: Làm tối ảnh (tăng cường các vùng sáng)")
