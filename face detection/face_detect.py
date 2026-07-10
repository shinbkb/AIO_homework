
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# PHẦN 1: PHÁT HIỆN KHUÔN MẶT TRÊN ẢNH TĨNH

def detect_face_on_image(image_path='anhcauthu.jpg'):
   
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không tìm thấy ảnh: {image_path}")
        return

    plt.imshow(image[:, :, ::-1])
    plt.axis('off')
    plt.show()

    # Khởi tạo bộ phân loại Haar Cascade
    facecascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

    for (a, b, w, h) in face_coordinates:
        cv2.rectangle(image, (a, b), (a + w, b + h), (255, 0, 0), 2)

    print("face detection's coordinate: ", face_coordinates)
    cv2.imshow('frames', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# PHẦN 2: THU THẬP DỮ LIỆU KHUÔN MẶT QUA WEBCAM

def collect_face_data(num_samples=10):
   
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
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)
            for (a, b, w, h) in face_coordinates:
                # Cắt vùng mặt → resize 50x50 → flatten thành vector 2500 chiều
                face_crop    = gray[b:b + h, a:a + w]
                face_resized = cv2.resize(face_crop, (50, 50))
                face_data.append(face_resized.flatten())
                i += 1
                # Vẽ box + đếm số ảnh đã thu
                cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} [{i}/{num_samples}]",
                            (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                if i >= num_samples:   # đủ mẫu → dừng
                    ret = False
                    break
            cv2.imshow('frames', frame)
            if cv2.waitKey(1) == 27:   # ESC thoát sớm
                break
        else:
            print('error')
            break

    cv2.destroyAllWindows()
    camera.release()

    # Lưu dữ liệu
    face_data = np.asarray(face_data)
    face_data = face_data.reshape(num_samples, -1)   # shape: (num_samples, 2500)

    # faces.pkl
    faces_path = os.path.join(DATASET_DIR, 'faces.pkl')
    if not os.path.exists(faces_path):
        with open(faces_path, 'wb') as f:
            pickle.dump(face_data, f)
    else:
        with open(faces_path, 'rb') as f:
            existing_faces = pickle.load(f)
        face_data = np.append(existing_faces, face_data, axis=0)
        with open(faces_path, 'wb') as f:
            pickle.dump(face_data, f)

    # names.pkl
    names = [name] * num_samples
    names_path = os.path.join(DATASET_DIR, 'names.pkl')
    if not os.path.exists(names_path):
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(names_path, 'rb') as f:
            existing_names = pickle.load(f)
        names = existing_names + names
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)

    print("Done! Saved to:", DATASET_DIR)
    return face_data, names

# PHẦN 3: THUẬT TOÁN KNN (tự cài đặt)

def knn_predict(x_train, y_train, x_test, k=5):
    distances = np.sqrt(np.sum((x_train - x_test) ** 2, axis=1))
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels  = [y_train[i] for i in k_nearest_indices]
    label, count = np.unique(k_nearest_labels, return_counts=True)
    return label[np.argmax(count)]


def load_dataset():
    #Load faces.pkl và names.pkl từ thư mục dataset.
    DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
    with open(os.path.join(DATASET_DIR, 'faces.pkl'), 'rb') as f:
        faces = pickle.load(f)
    with open(os.path.join(DATASET_DIR, 'names.pkl'), 'rb') as f:
        labels = pickle.load(f)
    print('Shape of Faces matrix --> ', faces.shape)
    print('Labels:', labels)
    return faces, labels


def run_knn_recognition(faces, labels, k=5):
    #Nhận diện khuôn mặt thời gian thực bằng KNN.
    facecascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
            for (a, b, w, h) in face_coordinates:
                # Cắt vùng mặt → resize 50x50 → flatten
                fc   = gray[b:b + h, a:a + w]
                r    = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
                # Dự đoán bằng KNN
                text = knn_predict(faces, labels, r[0], k=k)
                cv2.putText(frame, text, (a, b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)
            cv2.imshow('livetime face recognition (KNN)', frame)
            if cv2.waitKey(1) == 27:   # ESC để thoát
                break
        else:
            print('error')
            break
    cv2.destroyAllWindows()
    camera.release()



# PHẦN 4: THUẬT TOÁN SVM

def svm_fit(x_train, y_train, kernel='linear', C=1.0):
    svm = SVC(kernel=kernel, C=C)
    svm.fit(x_train, y_train)
    return svm


def svm_predict(model, x_test):
    return model.predict(x_test.reshape(1, -1))[0]


def run_svm_recognition(faces, labels):
    #Nhận diện khuôn mặt thời gian thực bằng SVM.
    svm_model = svm_fit(faces, labels)

    facecascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
            for (a, b, w, h) in face_coordinates:
                fc   = gray[b:b + h, a:a + w]
                r    = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
                text = svm_predict(svm_model, r[0])
                cv2.putText(frame, text, (a, b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)
            cv2.imshow('livetime face recognition (SVM)', frame)
            if cv2.waitKey(1) == 27:   # ESC để thoát
                break
        else:
            print("error")
            break
    cv2.destroyAllWindows()
    camera.release()


# PHẦN 5: THUẬT TOÁN DECISION TREE

def dt_fit(x_train, y_train, max_depth=None):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    tree_clf.fit(x_train, y_train)
    return tree_clf


def dt_predict(model, x_test):
    return model.predict(x_test.reshape(1, -1))[0]


def run_dt_recognition(faces, labels):
    tree_clf = dt_fit(faces, labels)

    facecascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)
            for (a, b, w, h) in face_coordinates:
                fc   = gray[b:b + h, a:a + w]   # ảnh xám → 2500 features
                r    = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
                text = dt_predict(tree_clf, r[0])
                cv2.putText(frame, text, (a, b - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)
            cv2.imshow('livetime face recognition (DT)', frame)
            if cv2.waitKey(1) == 27:   # ESC để thoát
                break
        else:
            print("error")
            break
    cv2.destroyAllWindows()
    camera.release()



if __name__ == '__main__':
    DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
    faces_path  = os.path.join(DATASET_DIR, 'faces.pkl')

    print("=" * 45)
    print("  NHẬN DIỆN KHUÔN MẶT — face_detect.py")
    print("=" * 45)
    print("\nChọn chế độ:")
    print("  0 - Thu thập dữ liệu khuôn mặt qua webcam")
    print("  1 - Nhận diện bằng KNN")
    print("  2 - Nhận diện bằng SVM")
    print("  3 - Nhận diện bằng Decision Tree")
    mode = input("\nNhập lựa chọn (0/1/2/3): ").strip()

    if mode == '0':
        collect_face_data(num_samples=10)
    elif mode in ('1', '2', '3'):
        if not os.path.exists(faces_path):
            print("\n[LỖI] Chưa có dataset!")
            print("  Hãy chạy lại và chọn '0' để thu thập dữ liệu trước.")
        else:
            faces, labels = load_dataset()
            if mode == '1':
                run_knn_recognition(faces, labels, k=5)
            elif mode == '2':
                run_svm_recognition(faces, labels)
            elif mode == '3':
                run_dt_recognition(faces, labels)
    else:
        print("Lựa chọn không hợp lệ.")
