from utils.tools import *

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
# [PCDH] epoch:720, bit:48, dataset:nuswide_21, MAP:0.653, Best MAP: 0.659
# [PCDH] epoch:1785, bit:48, dataset:cifar10-1, MAP:0.166, Best MAP: 0.168

def get_config():
    config = {
        "alpha": 1,
        "beta": 1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[PCDH]",
        "resize_size": 144,
        "crop_size": 128,
        "batch_size": 64,
        "net": Net,
        # "dataset": "cifar10-1",
        "dataset": "nuswide_21",
        "epoch": 2000,
        "test_map": 15,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class Net(nn.Module):
    def __init__(self, hash_bit, num_classes, pretrained=True):
        super(Net, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.feature_layer = nn.Linear(8 * 8 * 256, 1024)
        self.hash_like_layer = nn.Sequential(nn.Linear(1024, hash_bit), nn.Tanh())
        self.discrete_hash_layer = nn.Linear(hash_bit, hash_bit)
        self.classification_layer = nn.Linear(hash_bit, num_classes, bias=False)

    def forward(self, x, istraining=False):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        feature = self.feature_layer(x)
        h = self.hash_like_layer(feature)
        b = self.discrete_hash_layer(h).add(1).mul(0.5).clamp(min=0, max=1)
        b = (b >= 0.5).float() * 2 - 1
        y_pre = self.classification_layer(b)
        if istraining:
            return feature, h, y_pre
        else:
            return b


class PCDHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(PCDHLoss, self).__init__()
        self.m = 2 * bit

    def forward(self, feature, h, y_pre, y, ind, config):

        dist = (h.unsqueeze(1) - h.unsqueeze(0)).pow(2).sum(dim=2)
        s = (y @ y.t() == 0).float()

        loss1 = (1 - s) / 2 * dist + s / 2 * (self.m - dist).clamp(min=0).pow(2)
        loss1 = loss1.mean()

        dist2 = (feature.unsqueeze(1) - feature.unsqueeze(0)).pow(2).sum(dim=2)
        loss2 = (1 - s) / 2 * dist2 + s / 2 * (self.m - dist2).clamp(min=0).pow(2)
        loss2 = loss2.mean()

        if "nuswide" in config["dataset"]:
            Lc = (y_pre - y * y_pre + ((1 + (-y_pre).exp()).log())).sum(dim=1).mean()
        else:
            Lc = (-y_pre.softmax(dim=1).log() * y).sum(dim=1).mean()

        return loss1 + config["alpha"] * loss2 + config["beta"] * Lc


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, config["n_class"]).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = PCDHLoss(config, bit)

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
            feature, h, y_pre = net(image, istraining=True)

            loss = criterion(feature, h, y_pre, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
