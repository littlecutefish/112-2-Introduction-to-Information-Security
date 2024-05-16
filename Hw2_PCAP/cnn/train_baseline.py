#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script to fetch the required dataset
@author : Anant
"""
import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import CNN

# 修改您的資料集路徑
train_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\train"
val_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\val"
test_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\test"

def custom_dataset_loader(train_dir, val_dir, test_dir, batch_size):
    """
    Loads the custom dataset into 3 sets: train, validation, and test
    :param train_dir: directory containing training data
    :param val_dir: directory containing validation data
    :param test_dir: directory containing test data
    :param val_split: a float value to decide the train and validation set split
    :param batch_size: an int value defining the batch size of the dataset
    :return train_dataloader: a PyTorch dataloader iterator for the training set
    :return val_dataloader: a PyTorch dataloader iterator for the validation set
    :return test_dataloader: a PyTorch dataloader iterator for the test set
    """
    transform = transforms.Compose([
        transforms.Grayscale(1),  # 將圖像轉換為灰度格式
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 將圖像轉換為張量
        transforms.Normalize([0.485, ], [0.229, ])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Define dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("-" * 30 + "CUSTOM DATASET" + "-" * 30)
    print("Train Set size: ", len(train_dataset))
    print("Validation Set size: ", len(val_dataset))
    print("Test Set size: ", len(test_dataset))

    return train_dataloader, val_dataloader, test_dataloader

def train(model, device, dataloader, criterion, optimizer):
    """
    Trains a baseline model for the given dataset
    :param model: a CNN model required for training
    :param device: the device used to train the model - GPU/CPU
    :param dataloader: training data iterator used to train the model
    :param criterion: criterion used to calculate the traninig loss
    :param optimzer: Optimzer used to update the model parameters using backpropagation
    :return train_loss: training loss for the current epoch
    """
    train_loss = 0.0
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
        data, target = data.to(device), target.to(device)
        # set gradients to zero
        optimizer.zero_grad()
        # Get output prediction from the model
        output = model(data)
        # Computer loss
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        # Collect new set of gradients
        loss.backward()
        # Upadate the model
        optimizer.step()

    return train_loss / len(dataloader.dataset)

def test(model, dataloader, criterion):
    """
    Tests the baseline model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :param criterion: criterion used to calculate the test loss
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """
    test_loss = 0.0
    correct = 0
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)

    return test_loss / len(dataloader.dataset), preds, accuracy

if __name__ == "__main__":
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', DEVICE)

    # 加載資料集
    train_data, validation_data, test_data = custom_dataset_loader(train_dir, val_dir, test_dir, batch_size=BATCH_SIZE)

    # 定義模型
    model = CNN().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log_directory = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated"
    logname = os.path.join(log_directory, 'log_baseline_custom_dataset_' + str(NUM_EPOCHS) + '.log')

    # 檢查日誌目錄是否存在，如果不存在則創建它
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 初始化 logger
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()

    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    # 设置保存模型的路径
    save_path = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated\models\custom_dataset_baseline.sav"

    # 检查目录是否存在，如果不存在则创建
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 訓練模型
    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch :", str(epoch))
        # 使用訓練數據進行訓練
        train_loss = train(model, DEVICE, train_data, criterion, optimizer)
        # 在驗證數據上進行測試
        val_loss, _, accuracy = test(model, validation_data, criterion)
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        logger.info('Epoch: {}/{}, Train Loss: {:.2f}, Val Loss: {:.2f}, Val Accuracy: {:.2f}%'.format(epoch, NUM_EPOCHS,
                                                                                                      train_loss,
                                                                                                      val_loss,
                                                                                                      accuracy * 100))
        print('Epoch: {}/{}, Train Loss: {:.2f}, Val Loss: {:.2f}, Val Accuracy: {:.2f}%'.format(epoch, NUM_EPOCHS,
                                                                                                      train_loss,
                                                                                                      val_loss,
                                                                                                      accuracy * 100))

        # 如果驗證損失減少，保存模型
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(model.state_dict(), save_path)

    # 從訓練中加載最佳模型
    # 加载模型
    model.load_state_dict(
        torch.load(r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated\models\custom_dataset_baseline.sav"))
    # 使用測試數據測試模型
    test_loss, predictions, accuracy = test(model, test_data, criterion)
    logger.info('Test accuracy {:.2f}'.format(accuracy))
    print('Test accuracy {:.2f}%'.format(accuracy * 100))