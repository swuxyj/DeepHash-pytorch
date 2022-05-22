from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)
# code [CV_Project](https://github.com/aarathimuppalla/CV_Project)
# code [DSH_tensorflow](https://github.com/yg33717/DSH_tensorflow)

def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
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
        "epoch": 250,
        "test_map": 15,
        "save_path": "save/DSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.abs()).abs().mean()

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DSHLoss(config, bit)

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
        config["pr_curve_path"] = f"log/alexnet/DSH_{config['dataset']}_{bit}.json"
        train_val(config, bit)
