import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đọc ảnh
img = cv2.imread('doraemon.png', 0)  # Đọc ảnh dưới dạng grayscale

# Kiểm tra xem ảnh có được đọc thành công không
if img is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn!")
else:
    # Áp dụng Sobel theo hướng X
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    
    # Áp dụng Sobel theo hướng Y
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Tính gradient tổng hợp (magnitude)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Chuyển đổi về uint8 để hiển thị
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    sobel_combined = np.uint8(sobel_combined)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 5))
    
    # Ảnh gốc
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')
    
    # Sobel X
    plt.subplot(1, 4, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X (Cạnh dọc)')
    plt.axis('off')
    
    # Sobel Y
    plt.subplot(1, 4, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y (Cạnh ngang)')
    plt.axis('off')
    
    # Sobel tổng hợp
    plt.subplot(1, 4, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel tổng hợp')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Lưu kết quả
    cv2.imwrite('sobel_x.png', sobel_x)
    cv2.imwrite('sobel_y.png', sobel_y)
    cv2.imwrite('sobel_combined.png', sobel_combined)
    print("Đã lưu các ảnh kết quả!")
