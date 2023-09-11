from torchvision import models as models
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_len, num_classes):
        super(Model, self).__init__()
        self.model = models.resnet50(num_classes=num_classes)
        self.model.conv1 = torch.nn.Conv2d(input_len, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        ##Weights init
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        if not self.training:
            x = self.softmax(x)
        return x
