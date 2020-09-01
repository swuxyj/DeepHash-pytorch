from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from itertools import product
from random import shuffle
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


# CNNH(AAAI2014)
# paper [Supervised Hashing for Image Retrieval via Image Representation Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/download/8137/8861)
# code[CNNH-pytorch](https://github.com/heheqianqian/CNNH)
# [CNNH] epoch:20, bit:48, dataset:cifar10-1, MAP:0.134, Best MAP: 0.134
# [CNNH] epoch:80, bit:48, dataset:nuswide_21, MAP:0.386, Best MAP: 0.386


def get_config():
    config = {
        "T": 10,
        "H_save_path": "save/CNNH/",
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "betas": (0.9, 0.999)}},
        "info": "[CNNH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10-1",
        "dataset": "nuswide_21",
        "epoch": 150,
        "test_map": 10,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class CNNHLoss(torch.nn.Module):
    def __init__(self, config, train_labels, bit):

        super(CNNHLoss, self).__init__()
        S = (train_labels @ train_labels.t() > 0).float() * 2 - 1
        # load H if exists
        save_full_path = "%sH_T(%d)_bit(%d)_dataset(%s).pt" % (
            config["H_save_path"], config["T"], bit, config["dataset"])
        if os.path.exists(save_full_path):
            print("loading ", save_full_path)
            self.H = torch.load(save_full_path).to(config["device"])
        else:
            self.H = self.stage_one(config["num_train"], bit, config["T"], S, config["H_save_path"], config["dataset"],
                                    config["device"])

    def stage_one(self, n, q, T, S, H_save_path, dataset, device):

        if not os.path.exists(H_save_path):
            os.makedirs(H_save_path)

        H = 2 * torch.rand((n, q)).to(device) - 1
        L = H @ H.t() - q * S
        permutation = list(product(range(n), range(q)))
        for t in range(T):
            H_temp = H.clone()
            L_temp = L.clone()
            shuffle(permutation)
            for i, j in tqdm(permutation):
                # formula 7
                g_prime_Hij = 4 * L[i, :] @ H[:, j]
                g_prime_prime_Hij = 4 * (H[:, j].t() @ H[:, j] + H[i, j].pow(2) + L[i, i])
                # formula 6
                d = (-g_prime_Hij / g_prime_prime_Hij).clamp(min=-1 - H[i, j], max=1 - H[i, j])
                # formula 8
                L[i, :] = L[i, :] + d * H[:, j].t()
                L[:, i] = L[:, i] + d * H[:, j]
                L[i, i] = L[i, i] + d * d

                H[i, j] = H[i, j] + d

            if L.pow(2).mean() >= L_temp.pow(2).mean():
                H = H_temp
                L = L_temp
            save_full_path = "%sH_T(%d)_bit(%d)_dataset(%s).pt" % (H_save_path, t + 1, bit, dataset)
            torch.save(H.sign().cpu(), save_full_path)
            print("[CNNH stage 1][%d/%d] reconstruction loss:%.7f ,H save in %s" % (
                t + 1, T, L.pow(2).mean().item(), save_full_path))
        return H.sign()

    def forward(self, u, y, ind, config):
        loss = (u - self.H[ind]).pow(2).mean()
        return loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    # get database_labels
    clses = []
    for _, cls, _ in tqdm(train_loader):
        clses.append(cls)
    train_labels = torch.cat(clses).to(device).float()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    print("Stage 1: learning approximate hash codes.")
    criterion = CNNHLoss(config, train_labels, bit)
    print("Stage 2: learning images feature representation and hash functions.")

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
