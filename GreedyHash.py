from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 30,
                      "optim_params": {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}},

        # "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30,
        #               "optim_params": {"lr": 5e-5, "weight_decay": 5e-4}},

        "info": "[GreedyHash]",
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
        "epoch": 200,
        "test_map": 3,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    if config["dataset"] == "imagenet":
        config["alpha"] = 1
        config["optimizer"]["epoch_lr_decrease"] = 80
    return config


class GreedyHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])
        self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])

    def forward(self, u, onehot_y, ind, config):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        loss2 = config["alpha"] * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = GreedyHashLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

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
