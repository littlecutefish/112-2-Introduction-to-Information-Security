import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import os
import cv2

# 載入類別索引
with open(r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\cnn\class_indices.json") as f:
    class_indices = json.load(f)

# 資料路徑
train_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\train"
val_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\val"

# 影像資料增強
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 建立訓練和驗證資料集
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # 調整影像大小為 128x128
    batch_size=8,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),  # 調整影像大小為 128x128
    batch_size=8,
    class_mode='categorical'
)

# 建立CNN模型
model = Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),  # 明確定義輸入形狀
    Conv2D(8, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(class_indices), activation='softmax')
])

# 編譯模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator
)

# 儲存模型
model.save(r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\cnn\malware_cnn_model.h5")

# 印出每個epoch的結果
for epoch in range(len(history.history['loss'])):
    print(f"Epoch {epoch + 1}: "
          f"Loss = {history.history['loss'][epoch]:.4f}, "
          f"Accuracy = {history.history['accuracy'][epoch]:.4f}, "
          f"Val Loss = {history.history['val_loss'][epoch]:.4f}, "
          f"Val Accuracy = {history.history['val_accuracy'][epoch]:.4f}")
