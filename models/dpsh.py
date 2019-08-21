import torch.nn as nn
import torch
from torchvision import models

# https://github.com/jiangqy/DPSH-pytorch/blob/master/CNN_model.py
class AlexNet(nn.Module):
  def __init__(self, hash_bit):
    super(AlexNet, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    cl1 = nn.Linear(256 * 6 * 6, 4096)
    cl1.weight = model_alexnet.classifier[1].weight
    cl1.bias = model_alexnet.classifier[1].bias

    cl2 = nn.Linear(4096, 4096)
    cl2.weight = model_alexnet.classifier[4].weight
    cl2.bias = model_alexnet.classifier[4].bias

    self.classifier = nn.Sequential(
        nn.Dropout(),
        cl1,
        nn.ReLU(inplace=True),
        nn.Dropout(),
        cl2,
        nn.ReLU(inplace=True),
        nn.Linear(4096, hash_bit),
    )


  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x