
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing import image

# 載入類別索引
with open(r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\cnn\class_indices.json") as f:
    class_indices = json.load(f)

# 載入模型
model = tf.keras.models.load_model(r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\cnn\malware_cnn_model.h5")

# 圖片路徑
test_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\test"

# 預測圖片並計算準確度
total_images = 0
correct_predictions = 0

for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = class_indices[str(predicted_class_idx)]

            total_images += 1
            if predicted_class == class_name:
                correct_predictions += 1
                print(f"預測正確：圖片 '{img_name}' 的預測類別為 '{predicted_class}'。")
            else:
                print(f"預測錯誤：圖片 '{img_name}' 的預測類別為 '{predicted_class}'（真實類別為 '{class_name}'）。")

# 計算並打印準確度
accuracy = correct_predictions / total_images * 100
print(f"\n總共預測圖片數量: {total_images}")
print(f"正確預測數量: {correct_predictions}")
print(f"準確度: {accuracy:.2f}%")