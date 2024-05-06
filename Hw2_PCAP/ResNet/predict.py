import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34


def predict_images(test_dir, model, class_indict, device):
    correct = 0
    total = 0

    # 遍歷測試數據集中的所有類別文件夾
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        # 檢查是否是一個目錄
        if not os.path.isdir(class_dir):
            continue
        # 遍歷類別文件夾中的所有圖像文件
        for img_name in os.listdir(class_dir):
            # 檢查文件擴展名是否為 'png'
            if not img_name.endswith('.png'):
                continue  # 跳過非 '.png' 文件
            img_path = os.path.join(class_dir, img_name)
            # 載入圖像
            img = Image.open(img_path)
            # 圖像預處理
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(device)

            # 進行預測
            model.eval()
            with torch.no_grad():
                output = torch.squeeze(model(img))
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).item()

            # 獲取預測結果對應的類別名
            predicted_class = class_indict[str(predict_cla)]

            # 獲取圖像文件名中的類別名
            true_class = class_name

            # 比較預測結果和實際標籤
            if true_class == predicted_class:
                correct += 1
                print(f"預測正確：圖片 '{img_name}' 的預測類別為 '{predicted_class}'。")
            else:
                print(
                    f"預測錯誤：圖片 '{img_name}' 的預測類別為 '{predicted_class}'（真實類別為 '{true_class}'）。")

            total += 1

    # 計算準確率
    accuracy = correct / total * 100
    return accuracy


if __name__ == '__main__':
    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 數據預處理
    data_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ])
    ])

    # 載入類別字典
    with open('class_indices.json', "r") as f:
        class_indict = json.load(f)

    # 創建模型
    model = resnet34(num_classes=len(class_indict)).to(device)

    # 加載模型權重
    weights_path = "resNet34-v1.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 測試數據集路徑
    test_dir = r"C:\Users\User\Desktop\Malware\split_dataset\test"

    # 進行預測並計算準確率
    accuracy = predict_images(test_dir, model, class_indict, device)

    print("準確率:", accuracy)
