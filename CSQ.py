from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random

torch.multiprocessing.set_sharing_strategy('file_system')


# CSQ(CVPR2020)
# paper [Central Similarity Quantization for Efficient Image and Video Retrieval](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf)
# code [CSQ-pytorch](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)

# AlexNet
# [CSQ] epoch:65, bit:64, dataset:cifar10-1, MAP:0.787, Best MAP: 0.790
# [CSQ] epoch:90, bit:16, dataset:imagenet, MAP:0.593, Best MAP: 0.596, paper:0.601
# [CSQ] epoch:150, bit:64, dataset:imagenet, MAP:0.698, Best MAP: 0.706, paper:0.695
# [CSQ] epoch:40, bit:16, dataset:nuswide_21, MAP:0.784, Best MAP: 0.789
# [CSQ] epoch:40, bit:32, dataset:nuswide_21, MAP:0.821, Best MAP: 0.821
# [CSQ] epoch:40, bit:64, dataset:nuswide_21, MAP:0.834, Best MAP: 0.834

# ResNet50
# [CSQ] epoch:20, bit:64, dataset:imagenet, MAP:0.881, Best MAP: 0.881, paper:0.873
# [CSQ] epoch:10, bit:64, dataset:nuswide_21_m, MAP:0.844, Best MAP: 0.844, paper:0.839
# [CSQ] epoch:40, bit:64, dataset:coco, MAP:0.870, Best MAP: 0.883, paper:0.861
def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        # "net": AlexNet,
        "net": ResNet,
        # "dataset": "cifar10-1",
        "dataset": "imagenet",
        # "dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        "epoch": 150,
        "test_map": 10,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [64],
    }
    config = config_dataset(config)
    return config


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = CSQLoss(config, bit)

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
        config["pr_curve_path"] = f"log/alexnet/CSQ_{config['dataset']}_{bit}.json"
        train_val(config, bit)
