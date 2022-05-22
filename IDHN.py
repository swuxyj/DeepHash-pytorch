from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# IDHN(TMM2019)
# paper [Improved Deep Hashing with Soft Pairwise Similarity for Multi-label Image Retrieval](https://arxiv.org/abs/1803.02987)
# code [IDHN-Tensorflow](https://github.com/pectinid16/IDHN)

# [IDHN] epoch:75, bit:48, dataset:mirflickr, MAP:0.740, Best MAP: 0.740

def get_config():
    config = {
        "alpha": 0.5,
        "gamma": 0.1,
        "lambda": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[IDHN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 150,
        "test_map": 15,
        "save_path": "save/IDHN",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        self.q = bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u / (u.abs() + 1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = y @ self.Y.t()
        norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        s = s / (norm + 0.00001)

        M = (s > 0.99).float() + (s < 0.01).float()

        inner_product = config["alpha"] * u @ self.U.t()

        log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product

        mse_loss = (inner_product + self.q - 2 * s * self.q).pow(2)

        loss1 = (M * log_loss + config["gamma"] * (1 - M) * mse_loss).mean()
        loss2 = config["lambda"] * (u.abs() - 1).abs().mean()

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DPSHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
