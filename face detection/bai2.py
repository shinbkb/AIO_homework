import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)
print(f"Dataset sẽ lưu tại: {DATASET_DIR}")
face_data = []
i = 0
camera = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
name = input('Enter your name --> ')
ret = True
while ret:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)
        for (a, b, w, h) in face_coordinates:
            # Cắt vùng mặt → resize 50x50 → flatten thành vector 2500 chiều
            face_crop    = gray[b:b+h, a:a+w]
            face_resized = cv2.resize(face_crop, (50, 50))
            face_data.append(face_resized.flatten())
            i += 1
            # Vẽ box + đếm số ảnh đã thu
            cv2.rectangle(frame, (a, b), (a+w, b+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} [{i}/10]",
                        (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if i >= 10:          # đủ 10 ảnh → dừng
                ret = False
                break
        cv2.imshow('frames', frame)
        if cv2.waitKey(1) == 27:  # ESC để thoát sớm
            break
    else:
        print('error')
        break
cv2.destroyAllWindows()
camera.release()
face_data = np.asarray(face_data)
face_data = face_data.reshape(10, -1)    # shape: (10, 2500)
if 'faces.pkl' not in os.listdir(DATASET_DIR):
    with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'wb') as f:
        pickle.dump(faces, f)
names = [name] * 10
if 'names.pkl' not in os.listdir(DATASET_DIR):
    with open(os.path.join(DATASET_DIR, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
else:
    with open(os.path.join(DATASET_DIR, 'names.pkl'), 'rb') as f:
        existing_names = pickle.load(f)
    names = existing_names + names
    with open(os.path.join(DATASET_DIR, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
print("Done! Saved to:", DATASET_DIR)