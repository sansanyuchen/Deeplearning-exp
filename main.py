#导入必要的包
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
def train_model(model, dataloader , criterion , optimizer , num_epochs , device,val_loader,checkpoint_dir,scheduler):
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()
        train_loss = running_loss / len(dataloader.dataset)
        history["train_loss"].append( train_loss)
        if epoch %10 ==0 :
            save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir)
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = 100. * correct / total
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f},accuracy:{accuracy:.7f}')
    return model, history
def test_model(model, test_loader, device,criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
if __name__ == '__main__':
    batch_size = 2048
    print("train start")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    model = MyNet(200)
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = 0.9
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    epoch = 50
    device = torch.device("cuda:0")
    checkpoint_dir = 'logs'
    _, history = train_model(model, dataloader=train_dataloader, criterion=criterion, optimizer=optimizer, num_epochs=epoch,
                device=device, val_loader=val_dataloader, checkpoint_dir=checkpoint_dir,scheduler=scheduler)
    plt.figure(figsize=(8, 6))
    test_model(model = model , test_loader=test_dataloader, device = device,criterion=criterion)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='red', marker='o')
    plt.title('Loss Curves for Train and Validation', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.savefig('loss_curve.png')

