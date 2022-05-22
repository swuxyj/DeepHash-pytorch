from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# QSMIH(Signal Processing: Image Communication)
# paper [Deep supervised hashing using quadratic spherical mutual information for efficient image retrieval](https://www.sciencedirect.com/science/article/pii/S0923596521000072)
# code [qsmi  pytorch](https://github.com/passalis/qsmi)

# "net": AlexNet, "alpha": 0.001
# [QSMIH] epoch:150, bit:48, dataset:cifar10, MAP:0.777, Best MAP: 0.777
# [QSMIH] epoch:30, bit:48, dataset:nuswide_21, MAP:0.803, Best MAP: 0.811

# "net": AlexNet, "alpha": 0.01
# [QSMIH] epoch:45, bit:48, dataset:nuswide_21, MAP:0.821, Best MAP: 0.821
# [QSMIH] epoch:10, bit:48, dataset:coco, MAP:0.639, Best MAP: 0.639
# [QSMIH] epoch:70, bit:48, dataset:cifar10, MAP:0.762, Best MAP: 0.779
def get_config():
    config = {
        "alpha": 0.01,
        "sigma": 0,
        "use_square_clamp": True,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[QSMIH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 150,
        "test_map": 5,
        "save_path": "save/QSMIH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


# modify from  https://github.com/passalis/qsmi/blob/master/hashing/qmi_hashing.py
class QSMIHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(QSMIHLoss, self).__init__()

    def forward(self, u, y, ind, config):


        u = u / (torch.sqrt(torch.sum(u ** 2, dim=1, keepdim=True)) + 1e-8)
        Y = torch.mm(u, u.t())
        Y = 0.5 * (Y + 1)

        # Get the indicator matrix \Delta
        # D = (y.view(y.shape[0], 1) == y.view(1, y.shape[0]))
        D = (y @ y.t() > 0).float()

        M = D.size(1) ** 2 / torch.sum(D)

        if config["use_square_clamp"]:
            Q_in = (D * Y - 1) ** 2
            Q_btw = (1.0 / M) * Y ** 2
            # Minimize clamped loss
            L_QSMI = Q_in + Q_btw
        else:
            Q_in = D * Y
            Q_btw = (1.0 / M) * Y
            # Maximize QMI/QSMI
            L_QSMI = Q_btw - Q_in

        L_QSMI = L_QSMI.mean()
        L_hash = config["alpha"] * (u.abs() - 1).abs().mean()
        return L_QSMI + L_hash


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = QSMIHLoss(config, bit)

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
