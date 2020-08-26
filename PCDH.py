from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

# PCDH(Neurocomputing 2020)
# paper [Deep discrete hashing with pairwise correlation learning](https://www.sciencedirect.com/science/article/pii/S092523121931793X)

class AlexNet(nn.Module):
    def __init__(self, hash_bit, num_classes, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.feature_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
        )
        self.hashing_layer = nn.Sequential(nn.Linear(4096, hash_bit), nn.Tanh())
        self.layer1 = nn.Linear(hash_bit, hash_bit)
        self.layer2 = nn.Linear(hash_bit, num_classes)

    def forward(self, x, istraining=False):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        featrue = self.feature_layer(x)
        h = self.hashing_layer(featrue)
        b = self.layer1(h).add(1).mul(0.5).clamp(min=0, max=1)
        b = (b >= 0.5).float().mul(2).sub(1)
        wb = self.layer2(b)
        if istraining:
            return b
        else:
            return featrue, h, wb

