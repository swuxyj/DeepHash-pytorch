from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DTSH(ACCV2016)
# paper [Deep Supervised Hashing with Triplet Labels](https://arxiv.org/abs/1612.03900)
# code [DTSH](https://github.com/Minione/DTSH)

def get_config():
    config = {
        "alpha": 5,
        "lambda": 1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DTSH]",
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
        "save_path": "save/DTSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48, 32, 24, 12],
    }
    config = config_dataset(config)
    return config


class DTSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DTSHLoss, self).__init__()

    def forward(self, u, y, ind, config):

        inner_product = u @ u.t()
        s = y @ y.t() > 0
        count = 0

        loss1 = 0
        for row in range(s.shape[0]):
            # if has positive pairs and negative pairs
            if s[row].sum() != 0 and (~s[row]).sum() != 0:
                count += 1
                theta_positive = inner_product[row][s[row] == 1]
                theta_negative = inner_product[row][s[row] == 0]
                triple = (theta_positive.unsqueeze(1) - theta_negative.unsqueeze(0) - config["alpha"]).clamp(min=-100,
                                                                                                             max=50)
                loss1 += -(triple - torch.log(1 + torch.exp(triple))).mean()

        if count != 0:
            loss1 = loss1 / count
        else:
            loss1 = 0

        loss2 = config["lambda"] * (u - u.sign()).pow(2).mean()

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DTSHLoss(config, bit)

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
