import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import json

# 設定訓練參數
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# 檔案路徑
train_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\train"
val_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\val"
test_dir = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\split_dataset\test"
class_json_path = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\CNN\class_indices.json"

# 載入類別對應關係
with open(class_json_path, 'r') as f:
    class_dict = json.load(f)

# 資料轉換
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 載入資料集
train_data = ImageFolder(train_dir, transform=transform)
val_data = ImageFolder(val_dir, transform=transform)
test_data = ImageFolder(test_dir, transform=transform)

# 定義資料載入器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


# 定義CNN模型
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


# 初始化模型、損失函數和優化器
model = CNN(num_classes=len(class_dict))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 評估模型在驗證集上的表現
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total

    print('Epoch [{}/{}], Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, running_loss / len(train_loader), val_accuracy * 100))

# 測試模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
