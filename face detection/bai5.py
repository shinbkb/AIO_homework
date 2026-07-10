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
from sklearn.tree import DecisionTreeClassifier

def dt_fit(x_train, y_train, max_depth=None):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    tree_clf.fit(x_train, y_train)
    return tree_clf
def dt_predict(model, x_test):
    return model.predict(x_test.reshape(1, -1))[0]
tree_clf = dt_fit(faces, labels)
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
        for (a, b, w, h) in face_coordinates:
            fc   = gray[b:b + h, a:a + w]                        # ảnh xám → 2500 features
            r    = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            text = dt_predict(tree_clf, r[0])
            cv2.putText(frame, text, (a, b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)
        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:    # ESC để thoát
            break
    else:
        print("error")
        break
cv2.destroyAllWindows()
camera.release()