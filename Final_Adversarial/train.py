import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34


def fgsm_attack(image, epsilon, data_grad):
    # FGSM attack
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image


def fgsm_attack_v2(model, loss, images, labels, eps):
    images.requires_grad = True

    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()

    attack_images = images + eps * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        'train': transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, ], [0.229, ])
        ])
    }

    image_path = os.path.abspath(os.path.join(os.getcwd(), r"C:\Users\User\Downloads\MalImg dataset\dataset_9010\dataset_9010\malimg_dataset"))
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    malware_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in malware_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=6)
    with open('class_indices_new.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "validation"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = r"C:\Users\User\Desktop\Malware\Final_Adversarial\resNet34.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # 加载预训练模型并且把不需要的层去掉
    pre_state_dict = torch.load(model_weight_path)
    print("原模型", pre_state_dict.keys())
    new_state_dict = {}
    for k, v in net.state_dict().items():  # 遍历修改模型的各个层
        print("新模型", k)
        if k in pre_state_dict.keys() and k != 'conv1.weight':
            new_state_dict[k] = pre_state_dict[k]  # 如果原模型的层也在新模型的层里面， 那新模型就加载原先训练好的权重
    net.load_state_dict(new_state_dict, False)
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 6)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = r'C:\Users\User\Desktop\Malware\Final_Adversarial/resNet34_v5_no_attack.pth'
    train_steps = len(train_loader)
    epsilon = 0.5  # 攻擊強度

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # images, labels = images.to(device), labels.to(device)
            #
            # # 生成對抗性樣本
            # perturbed_data = fgsm_attack_v2(net, loss_function, images, labels, epsilon)

            # 用對抗性樣本進行訓練
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
