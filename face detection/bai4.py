import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# Load data từ bài tập 2
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'rb') as f:
    faces = pickle.load(f)
with open(os.path.join(DATASET_DIR, 'names.pkl'), 'rb') as f:
    labels = pickle.load(f)
from sklearn.svm import SVC

THRESHOLD = 0.7   # ngưỡng confidence: >= 0.7 → nhận diện, < 0.7 → Unknown

def svm_fit(x_train, y_train, kernel='linear', C=1.0):
    svm = SVC(kernel=kernel, C=C, probability=True)  # bật probability để lấy confidence
    svm.fit(x_train, y_train)
    return svm

def svm_predict(model, x_test, threshold=THRESHOLD):
    proba      = model.predict_proba(x_test.reshape(1, -1))[0]  # xác suất từng lớp
    confidence = np.max(proba)                                    # lấy xác suất cao nhất
    if confidence >= threshold:
        return model.predict(x_test.reshape(1, -1))[0], confidence
    else:
        return "Unknown", confidence

svm_model = svm_fit(faces, labels)
camera = cv2.VideoCapture(0)
# Face recognition using SVM
while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
        for (a, b, w, h) in face_coordinates:
            fc = gray[b:b + h, a:a + w]
            r  = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            # Dự đoán bằng SVM có ngưỡng confidence
            text, conf = svm_predict(svm_model, r[0])
            color = (0, 255, 0) if text != "Unknown" else (0, 0, 255)  # xanh/đỏ
            label_text = f"{text} ({conf:.0%})"
            cv2.putText(frame, label_text, (a, b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (a, b), (a + w, b + h), color, 2)
        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:    # ESC để thoát
            break
    else:
        print("error")
        break
cv2.destroyAllWindows()
camera.release()