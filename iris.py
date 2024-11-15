import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# === Phân lớp với dữ liệu IRIS ===
print("=== Phân lớp IRIS ===")
# Load dữ liệu IRIS
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# KNN trên dữ liệu IRIS
print("\n--- KNN (IRIS) ---")
knn = KNeighborsClassifier(n_neighbors=3)
start_time = time.time()
knn.fit(X_train_iris, y_train_iris)
knn_pred = knn.predict(X_test_iris)
knn_accuracy = accuracy_score(y_test_iris, knn_pred)
knn_time = time.time() - start_time
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print(f"KNN Time: {knn_time:.4f} seconds")

# SVM trên dữ liệu IRIS
print("\n--- SVM (IRIS) ---")
svm = SVC(kernel='linear')
start_time = time.time()
svm.fit(X_train_iris, y_train_iris)
svm_pred = svm.predict(X_test_iris)
svm_accuracy = accuracy_score(y_test_iris, svm_pred)
svm_time = time.time() - start_time
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"SVM Time: {svm_time:.4f} seconds")

# ANN trên dữ liệu IRIS
print("\n--- ANN (IRIS) ---")
y_train_iris_cat = to_categorical(y_train_iris)
y_test_iris_cat = to_categorical(y_test_iris)
ann = Sequential([
    Dense(16, input_shape=(X_train_iris.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
ann.fit(X_train_iris, y_train_iris_cat, epochs=50, batch_size=5, verbose=0)
ann_time = time.time() - start_time
ann_pred = ann.predict(X_test_iris)
ann_accuracy = accuracy_score(y_test_iris, np.argmax(ann_pred, axis=1))
print(f"ANN Accuracy: {ann_accuracy:.2f}")
print(f"ANN Time: {ann_time:.4f} seconds")

# === Phân lớp với dữ liệu CIFAR-10 (ảnh động vật) ===
print("\n=== Phân lớp CIFAR-10 ===")
# Load dữ liệu CIFAR-10
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
# Chỉ sử dụng 2 lớp "cat" và "dog" để phân loại ảnh động vật
animal_classes = [3, 5]  # 3: cat, 5: dog
train_filter = np.isin(y_train_cifar, animal_classes).flatten()
test_filter = np.isin(y_test_cifar, animal_classes).flatten()
X_train_animals, y_train_animals = X_train_cifar[train_filter], y_train_cifar[train_filter].flatten()
X_test_animals, y_test_animals = X_test_cifar[test_filter], y_test_cifar[test_filter].flatten()

# Chuẩn hóa dữ liệu
X_train_animals = X_train_animals.astype('float32') / 255
X_test_animals = X_test_animals.astype('float32') / 255

# Đổi nhãn thành 0 và 1 cho hai lớp
y_train_animals = np.where(y_train_animals == 3, 0, 1)
y_test_animals = np.where(y_test_animals == 3, 0, 1)

# ANN trên dữ liệu CIFAR-10 (ảnh động vật)
print("\n--- ANN (CIFAR-10) ---")
y_train_animals_cat = to_categorical(y_train_animals, 2)
y_test_animals_cat = to_categorical(y_test_animals, 2)
ann_cifar = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
ann_cifar.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
ann_cifar.fit(X_train_animals, y_train_animals_cat, epochs=10, batch_size=64, verbose=0)
ann_cifar_time = time.time() - start_time
ann_cifar_pred = ann_cifar.predict(X_test_animals)
ann_cifar_accuracy = accuracy_score(y_test_animals, np.argmax(ann_cifar_pred, axis=1))
print(f"ANN (CIFAR-10) Accuracy: {ann_cifar_accuracy:.2f}")
print(f"ANN (CIFAR-10) Time: {ann_cifar_time:.4f} seconds")
