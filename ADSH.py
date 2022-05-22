from utils.tools import *
from network import *
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# ADSH(AAAI2018)
# paper [Asymmetric Deep Supervised Hashing](https://cs.nju.edu.cn/lwj/paper/AAAI18_ADSH.pdf)
# code1 [ADSH matlab + pytorch](https://github.com/jiangqy/ADSH-AAAI2018)
# code2 [ADSH_pytorch](https://github.com/TreezzZ/ADSH_PyTorch)

def get_config():
    config = {
        "gamma": 200,
        "num_samples": 2000,
        "max_iter": 150,
        "epoch": 3,
        "test_map": 10,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.001, "weight_decay": 5e-4}},
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}},
        "info": "[ADSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        "dataset": "cifar10-1",
        # "dataset": "nuswide_21",
        "save_path": "save/ADSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    # if config["dataset"] == "nuswide_21":
    #     config["gamma"] = 0
    config = config_dataset(config)
    return config


def calc_sim(database_label, train_label):
    S = (database_label @ train_label.t() > 0).float()
    # soft constraint
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def train_val(config, bit):
    device = config["device"]
    num_samples = config["num_samples"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    # get database_labels
    clses = []
    for _, cls, _ in tqdm(dataset_loader):
        clses.append(cls)
    database_labels = torch.cat(clses).to(device).float()

    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    Best_mAP = 0

    V = torch.zeros((num_dataset, bit)).to(device)
    for iter in range(config["max_iter"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s,  training...." % (
            config["info"], iter + 1, config["max_iter"], current_time, bit, config["dataset"]), end="")

        net.train()

        # sampling and construct similarity matrix
        select_index = np.random.permutation(range(num_dataset))[0: num_samples]
        if "cifar" in config["dataset"]:
            train_loader.dataset.data = np.array(dataset_loader.dataset.data)[select_index]
            train_loader.dataset.targets = np.array(dataset_loader.dataset.targets)[select_index]
        else:
            train_loader.dataset.imgs = np.array(dataset_loader.dataset.imgs)[select_index].tolist()
        sample_label = database_labels[select_index]

        Sim = calc_sim(sample_label, database_labels)
        U = torch.zeros((num_samples, bit)).to(device)

        train_loss = 0
        for epoch in range(config["epoch"]):
            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device).float()
                net.zero_grad()
                S = calc_sim(label, database_labels)
                u = net(image)
                u = u.tanh()
                U[ind, :] = u.data

                square_loss = (u @ V.t() - bit * S).pow(2)
                quantization_loss = config["gamma"] * (V[select_index[ind]] - u).pow(2)
                loss = (square_loss.sum() + quantization_loss.sum()) / (num_train * u.size(0))

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

        train_loss = train_loss / len(train_loader) / epoch
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        # learning binary codes: discrete coding
        barU = torch.zeros((num_dataset, bit)).to(device)
        barU[select_index, :] = U
        # calculate Q
        Q = -2 * bit * Sim.t() @ U - 2 * config["gamma"] * barU
        for k in range(bit):
            sel_ind = np.setdiff1d([ii for ii in range(bit)], k)
            V_ = V[:, sel_ind]
            U_ = U[:, sel_ind]
            Uk = U[:, k]
            Qk = Q[:, k]
            # formula 10
            V[:, k] = -(2 * V_ @ (U_.t() @ Uk) + Qk).sign()

        if (iter + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

def main():
    config = get_config()
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/ADSH_{config['dataset']}_{bit}.json"
        train_val(config, bit)

if __name__ == "__main__":
    main()
