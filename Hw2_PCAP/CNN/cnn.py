import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import os

# 定義 PyTorch 模型
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定義分散式數據
class MyClientData(tff.simulation.datasets.TestClientData):
    def __init__(self, client_data):
        self.client_data = client_data

    def create_tf_dataset_for_client(self, client_id):
        return self.client_data[client_id]

    @property
    def client_ids(self):
        return list(self.client_data.keys())

# 加載數據
def load_data(base_folder, class_labels, num_clients):
    client_data = {}
    for i in range(num_clients):
        client_data[i] = load_data_for_client(base_folder, class_labels)
    return client_data

def load_data_for_client(folder, class_labels):
    data = []
    labels = []
    for folder_name in os.listdir(folder):
        folder_path = os.path.join(folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    file_path = os.path.join(folder_path, filename)
                    image = read_image(file_path)  # 自行實現讀取圖片的函數
                    data.append(image)
                    labels.append(class_labels[folder_name])
    return (np.array(data), np.array(labels))

# 將 PyTorch 模型轉換為 TensorFlow Federated 模型
def create_federated_model():
    keras_model = CNN()  # 創建 PyTorch 模型
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=[(tf.TensorSpec(shape=(None, 64, 64, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32))],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 定義聯邦迭代方法
def build_federated_algorithm():
    return tff.learning.build_federated_averaging_process(
        model_fn=create_federated_model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

# 加載數據集
base_folder = r'C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset_new'
class_json_path = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\CNN\class_indices.json"

num_clients = 3
client_data = load_data(base_folder, class_labels, num_clients)
federated_data = MyClientData(client_data)

# 創建迭代方法
iterative_process = build_federated_algorithm()

# 開始訓練
state = iterative_process.initialize()
num_rounds = 100  # 定義聯邦學習的總輪數
num_clients_per_round = 3  # 定義每輪訓練中的客戶端數量

for round_num in range(1, num_rounds + 1):
    selected_clients = np.random.choice(federated_data.client_ids, size=num_clients_per_round, replace=False)
    federated_train_data = [federated_data.create_tf_dataset_for_client(x) for x in selected_clients]
    state, metrics = iterative_process.next(state, federated_train_data)
    print('Round {:2d}, Metrics: {}'.format(round_num, metrics))
