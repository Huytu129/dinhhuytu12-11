import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score

# === Chuẩn bị dữ liệu CIFAR-10 cho CNN ===
print("=== Chuẩn bị Dữ liệu CIFAR-10 cho CNN ===")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Chuẩn hóa ảnh
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# === Mô hình CNN cho Nhận dạng Đối tượng ===
print("\n=== Huấn luyện và Đánh giá CNN ===")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện CNN
start_time = time.time()
cnn_model.fit(X_train, y_train_cat, epochs=10, batch_size=64, verbose=0)
cnn_train_time = time.time() - start_time

# Đánh giá CNN
start_time = time.time()
cnn_pred = np.argmax(cnn_model.predict(X_test), axis=1)
cnn_eval_time = time.time() - start_time
cnn_accuracy = accuracy_score(y_test, cnn_pred)
print(f"CNN Accuracy: {cnn_accuracy:.2f}")
print(f"CNN Training Time: {cnn_train_time:.4f} seconds")
print(f"CNN Evaluation Time: {cnn_eval_time:.4f} seconds")

# === Chuẩn bị Dữ liệu CIFAR-10 cho R-CNN (Faster R-CNN) ===
# Chuyển đổi dữ liệu thành định dạng của Pytorch
print("\n=== Đánh giá R-CNN (Faster R-CNN) ===")
X_test_torch = [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255 for img in X_test[:10]]

# Tải mô hình Faster R-CNN tiền huấn luyện
rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
rcnn_model.eval()

# Đánh giá Faster R-CNN trên dữ liệu ảnh
rcnn_start_time = time.time()
rcnn_predictions = []
for img in X_test_torch:
    with torch.no_grad():
        pred = rcnn_model([img])[0]
        rcnn_predictions.append(pred)

rcnn_eval_time = time.time() - rcnn_start_time
print(f"R-CNN (Faster R-CNN) Evaluation Time for 10 samples: {rcnn_eval_time:.4f} seconds")
