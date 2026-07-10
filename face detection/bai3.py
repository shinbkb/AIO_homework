import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
THRESHOLD = 0.7   # ngưỡng confidence: >= 0.7 → nhận diện, < 0.7 → Unknown

def knn_predict(x_train, y_train, x_test, k=5, threshold=THRESHOLD):
    distances = np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))  # Khoảng cách Euclidean
    k_nearest_indices = np.argsort(distances)[:k]                  # Lấy chỉ số k điểm gần nhất
    k_nearest_labels = [y_train[i] for i in k_nearest_indices]    # Nhãn của k điểm gần nhất
    label, count = np.unique(k_nearest_labels, return_counts=True)
    best_label  = label[np.argmax(count)]
    confidence  = np.max(count) / k   # tỉ lệ phiếu bầu: vd. 4/5 = 0.8
    if confidence >= threshold:
        return best_label, confidence
    else:
        return "Unknown", confidence
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')  # Load dữ liệu đã thu thập ở Bài tập 2
with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'rb') as f:
    faces = pickle.load(f)
with open(os.path.join(DATASET_DIR, 'names.pkl'), 'rb') as f:
    labels = pickle.load(f)
print('Shape of Faces matrix --> ', faces.shape)
print('Labels:', labels)
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
        for (a, b, w, h) in face_coordinates:
            # Cắt vùng mặt → resize 50x50 → flatten
            fc = gray[b:b + h, a:a + w]
            r  = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            # Dự đoán bằng KNN có ngưỡng confidence
            text, conf = knn_predict(faces, labels, r[0], k=5)
            color = (0, 255, 0) if text != "Unknown" else (0, 0, 255)  # xanh/đỏ
            label_text = f"{text} ({conf:.0%})"
            cv2.putText(frame, label_text, (a, b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (a, b), (a + w, b + h), color, 2)
        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:   # ESC để thoát
            break
    else:
        print('error')
        break
cv2.destroyAllWindows()
camera.release()