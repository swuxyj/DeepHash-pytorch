from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DSHSD(IEEE ACCESS 2019)
# paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
# [DSHSDSD] epoch:70, bit:48, dataset:cifar10-1, MAP:0.809, Best MAP: 0.809
# [DSHSDSD] epoch:250, bit:48, dataset:nuswide_21, MAP:0.809, Best MAP: 0.815
# [DSHSDSD] epoch:135, bit:48, dataset:imagenet, MAP:0.647, Best MAP: 0.647
def get_config():
    config = {
        "alpha": 0.05,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "betas": (0.9, 0.999)}},
        "info": "[DSHSDSD]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10-1",
        "dataset": "imagenet",
        # "dataset": "nuswide_21",
        "epoch": 250,
        "test_map": 5,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DSHSDLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHSDLoss, self).__init__()
        self.m = 2 * bit
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])

    def forward(self, u, y, ind, config):
        u = torch.tanh(u)

        dist = (u.unsqueeze(1) - u.unsqueeze(0)).pow(2).sum(dim=2)
        s = (y @ y.t() == 0).float()

        Ld = (1 - s) / 2 * dist + s / 2 * (self.m - dist).clamp(min=0)
        Ld = config["alpha"] * Ld.mean()
        o = self.fc(u)
        if "nuswide" in config["dataset"]:
            # formula 8, multiple labels classification loss
            Lc = (o - y * o + ((1 + (-o).exp()).log())).sum(dim=1).mean()
        else:
            # formula 7, single labels classification loss
            Lc = (-o.softmax(dim=1).log() * y).sum(dim=1).mean()

        return Lc + Ld


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DSHSDLoss(config, bit)

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
