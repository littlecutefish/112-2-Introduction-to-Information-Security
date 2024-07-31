import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34
from opacus import PrivacyEngine

def fgsm_attack(image, epsilon, data_grad):
    # FGSM attack
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Clipping to [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def imshow(img, title=None):
    # Unnormalize and show the image
    img = img.cpu().detach().numpy().squeeze()
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')

def predict_images(test_dir, model, class_indict, device, epsilon, optimizer):
    adver_correct = 0
    origin_correct = 0
    dp_correct = 0
    total = 0

    origin_results = []
    adver_results = []
    dp_results = []

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

            # 確保模型在評估模式
            model.eval()
            img.requires_grad = True

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
                origin_correct += 1
                origin_results.append(f"預測正確：{predicted_class}")
            else:
                origin_results.append(f"預測錯誤：{predicted_class}（真實類別為 '{true_class}'）")

            # 初始預測
            output = model(img)
            init_pred = output.max(1, keepdim=True)[1]

            # 計算損失
            loss = torch.nn.CrossEntropyLoss()(output, init_pred.view(-1))

            # 計算梯度
            model.zero_grad()
            loss.backward()
            data_grad = img.grad.data

            # 生成對抗樣本
            perturbed_data = fgsm_attack(img, epsilon, data_grad)

            # 使用對抗樣本進行預測
            output = model(perturbed_data)
            predict = torch.softmax(output, dim=1)
            predict_cla = torch.argmax(predict, dim=1).item()

            # 獲取預測結果對應的類別名
            predicted_class = class_indict[str(predict_cla)]

            # 獲取圖像文件名中的類別名
            true_class = class_name

            # 比較預測結果和實際標籤
            if true_class == predicted_class:
                adver_correct += 1
                adver_results.append(f"預測正確：{predicted_class}")
            else:
                adver_results.append(f"預測錯誤：{predicted_class}（真實類別為 '{true_class}')")

            # # 差分隐私防御
            # model.train()
            # optimizer.zero_grad()
            # output = model(img)
            # loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([predict_cla], device=device))
            # loss.backward()
            # optimizer.step()
            #
            # output = model(img)
            # predict = torch.softmax(output, dim=1)
            # predict_cla = torch.argmax(predict, dim=1).item()
            # predicted_class = class_indict[str(predict_cla)]
            # if true_class == predicted_class:
            #     dp_correct += 1
            #     dp_results.append(f"預測正確：{predicted_class}")
            # else:
            #     dp_results.append(f"預測錯誤：{predicted_class}（真實類別為 '{true_class}')")
            #
            total += 1

    # 印出結果
    print("===== Original ======")
    for result in origin_results:
        print(result)
    print("===== Adversarial ======")
    for result in adver_results:
        print(result)
    # print("===== Differential Privacy Defense ======")
    # for result in dp_results:
    #     print(result)

    # 計算準確率
    origin_accuracy = origin_correct / total * 100
    adver_accuracy = adver_correct / total * 100
    # dp_accuracy = dp_correct / total * 100
    print("origin accuracy: ", origin_accuracy)
    print("adversarial accuracy: ", adver_accuracy)
    # print("dp accuracy: ", dp_accuracy)

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
    with open(r"C:\Users\User\Desktop\Malware\Final_Adversarial\class_indices_new.json", "r") as f:
        class_indict = json.load(f)

    # 創建模型
    model = resnet34(num_classes=len(class_indict)).to(device)

    # 加載模型權重
    weights_path = r"C:\Users\User\Desktop\Malware\Final_Adversarial\resNet34_v5_no_attack.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 測試數據集路徑
    test_dir = r"C:\Users\User\Downloads\MalImg dataset\dataset_9010\dataset_9010\malimg_dataset\test"

    # 設定對抗攻擊的擾動大小
    epsilon = 0.1

    # 進行預測並計算準確率
    predict_images(test_dir, model, class_indict, device, epsilon, optimizer)