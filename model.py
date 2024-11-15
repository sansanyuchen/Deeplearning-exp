import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, n_class):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),#64 31 31  0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), #2
            nn.MaxPool2d(kernel_size=3, stride=1),#64 29 29  #3
            nn.Conv2d(64, 192, kernel_size=5, padding=2),#192 30 30 # 4
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),#6
            nn.MaxPool2d(kernel_size=3, stride=2),#192 14 14 # 7
            nn.Conv2d(192, 384, kernel_size=3, padding=1),#384 14 14# 8
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True), # 10
            nn.Conv2d(384, 256, kernel_size=3, padding=1),#256 14 14 11
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),# 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),#256 1414  # 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),#  16
            nn.MaxPool2d(kernel_size=3, stride=2),#256 6 6  #17
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_class),
        )
        self.res1 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
    def forward(self, x):

        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        res1 = self.res1(x)
        x = self.features[8](x)
        res2 = self.res2(x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x) + res1
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x) + res2
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

'''
class MyNet(nn.Module):
    def __init__(self, n_class):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
'''

