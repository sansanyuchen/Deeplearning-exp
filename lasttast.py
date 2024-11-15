import random
import math
from cProfile import label
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
from tqdm import tqdm
from datasets import TinyImageDataset_test ,  TinyImageDataset_train
from model import  ResNet ,Bottleneck,CNN_net,MyNet
from utils import save_checkpoint
import torch.optim as optim

def test_model(model, test_loader, device):
    model.eval()  # 切换到评估模式
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)


            print(output.shape, target.shape)
            print(output,target)
            #loss = criterion(predicted , target)
            loss = criterion(output, target)

            #total_loss += loss.item()*data.size(0)

            # 计算准确率
            _, predicted = torch.max(output, 1)

            correct += (predicted == target).sum().item()
            print(correct)
            total += target.size(0)

    #avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    #print(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(f"Accuracy: {accuracy:.7f}%")
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 数据从 [0, 255] 映射到 [-1, 1]
    ])
    root_dir_train = 'tiny-imagenet-200/train'
    root_dir_val = 'tiny-imagenet-200/val'
    train_dataset = TinyImageDataset_train(root_dir_train, transform=transform)
    labels_name = train_dataset.get_label_name()
    test_dataset = TinyImageDataset_test(root_dir_val, labels_name, transform=transform)
    all_size = len(train_dataset)
    val_size = int(0.2 * all_size)  # 20% 用于验证集
    new_train_size = all_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [new_train_size, val_size])
    batch_size = 2000
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True,drop_last=True)
    checkpoint_path = ('checkpoint_epoch_40.pt')
    model = MyNet(200)
    device = torch.device("cuda:5" )
    # 加载模型时忽略不匹配的层
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"], strict=False)


    test_model(model, test_loader=test_dataloader, device = device)

