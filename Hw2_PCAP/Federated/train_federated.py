#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Script used to train the Federated Model
@author : Anant
"""
import os
import torch
import numpy as np
import logging.config
from tqdm import tqdm
from model import CNN
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import copy

NUM_EPOCHS = 10
LOCAL_ITERS = 2
BATCH_SIZE = 16
NUM_CLIENTS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)


# 修改您的資料集路徑
train_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\train"
val_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\val"
test_dir = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\split_dataset\test"

def custom_dataset_loader(train_dir, val_dir, test_dir, batch_size = DEVICE):
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
        transforms.Grayscale(num_output_channels=1),  # 將圖像轉換為灰度格式
        transforms.ToTensor()  # 將圖像轉換為張量
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

def FedAvg(params):
    """
    Average the paramters from each client to update the global model
    :param params: list of paramters from each client's model
    :return global_params: average of paramters from each client
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train(local_model, device, dataset, iters):
    """
    Trains a local model for a given client
    :param local_model: a copy of global CNN model required for training
    :param device: the device used to train the model - GPU/CPU
    :param dataset: training dataset used to train the model
    :return local_params: parameters from the trained model from the client
    :return train_loss: training loss for the current epoch
    """
    # optimzer for training the local models
    local_model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    train_loss = 0.0
    local_model.train()
    # Iterate for the given number of Client Iterations
    for i in range(iters):
        batch_loss = 0.0
        for batch_idx, (data, target) in tqdm(enumerate(dataset), total=len(dataset)):
            data, target = data.to(device), target.to(device)
            # set gradients to zero
            optimizer.zero_grad()
            # Get output prediction from the Client model
            output = local_model(data)
            # Computer loss
            loss = criterion(output, target)
            batch_loss += loss.item() * data.size(0)
            # Collect new set of gradients
            loss.backward()
            # Update local model
            optimizer.step()
        # add loss for each iteration
        train_loss += batch_loss / len(dataset)
    return local_model.state_dict(), train_loss / iters


def test(model, dataloader):
    """
    Tests the Federated global model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    model.eval()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader.dataset) / BATCH_SIZE):
        data, target = data, target
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
    accuracy = correct / len(dataloader.dataset)

    return test_loss / len(dataloader.dataset), preds, accuracy


if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')

    log_directory = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated"
    logname = os.path.join(log_directory, 'log_federated' + str(NUM_EPOCHS) + "_" + str(NUM_CLIENTS) + "_" + str(LOCAL_ITERS))

    # 檢查日誌目錄是否存在，如果不存在則創建它
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 初始化 logger
    logging.basicConfig(filename=logname, level=logging.DEBUG)
    logger = logging.getLogger()

    # 加載資料集
    train_data, validation_data, test_data = custom_dataset_loader(train_dir, val_dir, test_dir, batch_size=BATCH_SIZE)

    # 设置保存模型的路径
    save_path = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated\models\custom_dataset_federated.sav"

    # 检查目录是否存在，如果不存在则创建
    save_dir = os.path.dirname(save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # distribute the trainning data across clients
    train_distributed_dataset = [[] for _ in range(NUM_CLIENTS)]
    for batch_idx, (data, target) in enumerate(train_data):
        train_distributed_dataset[batch_idx % NUM_CLIENTS].append((data, target))

    # get model and define criterion for loss
    global_model = CNN()
    global_params = global_model.state_dict()

    global_model.train()
    all_train_loss = list()
    all_val_loss = list()
    val_loss_min = np.Inf

    # Train the model for given number of epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        print("\nEpoch :", str(epoch))
        local_params, local_losses = [], []
        # Send a copy of global model to each client
        for idx in range(NUM_CLIENTS):
            # Perform training on client side and get the parameters
            param, loss = train(copy.deepcopy(global_model), DEVICE, train_distributed_dataset[idx], LOCAL_ITERS)
            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(loss))

        # Federated Average for the paramters from each client
        global_params = FedAvg(local_params)
        # Update the global model
        global_model.load_state_dict(global_params)
        all_train_loss.append(sum(local_losses) / len(local_losses))

        # Test the global model
        val_loss, _, accuracy = test(global_model, validation_data)
        all_val_loss.append(val_loss)

        logger.info('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}' \
                    .format(epoch, NUM_EPOCHS, all_train_loss[-1], val_loss, accuracy))
        print('Epoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}' \
                    .format(epoch, NUM_EPOCHS, all_train_loss[-1], val_loss, accuracy))

        # if validation loss decreases, save the model
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            logger.info("Saving Model State")
            torch.save(global_model.state_dict(), save_path)

    # load the best model from training
    global_model.load_state_dict(torch.load(r"C:\Users\User\Desktop\Malware\Hw2_PCAP\Federated\models\custom_dataset_federated.sav"))
    # test the model using test data
    test_loss, predictions, accuracy = test(global_model, test_data)
    logger.info('Test accuracy {:.8f}'.format(accuracy))
    print('Test accuracy {:.2f}%'.format(accuracy*100))