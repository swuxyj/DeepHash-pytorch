from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DAPH(ACM International Conference on Multimedia (MM), 2017 )
# paper [Deep Asymmetric Pairwise Hashing](http://cfm.uestc.edu.cn/~fshen/DAPH.pdf)

def get_config():
    config = {
        "alpha": 10,
        "gamma": 10,
        "lambda": 0.01,
        "beta": 0.01,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.0001, "weight_decay": 0.0001}},
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}},
        "info": "[DAPH]",
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
        "test_map": 5,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DAPHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DAPHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Z = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        self.I = torch.eye(bit).to(config["device"])
        self.B = torch.randn(config["num_train"], bit).sign().to(config["device"])
        self.H = torch.randn(config["num_train"], bit).sign().to(config["device"])

    def forward(self, u, z, y, ind, config, isTop=1):
        u = u.tanh()
        z = z.tanh()
        self.U[ind, :] = u.data
        self.Z[ind, :] = z.data
        self.Y[ind, :] = y.float()

        s = (y @ y.t() > 0).float()
        inner_product = u @ z.t() * 0.5
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        quantization_loss = isTop * (u - self.B[ind]).pow(2) + (1 - isTop) * (z - self.H[ind]).pow(2)
        quantization_loss = config["alpha"] * quantization_loss.mean()

        independence_loss = isTop * (u.t() @ u / u.shape[0] - self.I).pow(2) + \
                            (1 - isTop) * (z.t() @ z / z.shape[0] - self.I).pow(2)
        independence_loss = config["lambda"] * independence_loss.mean()

        balance_loss = isTop * u.sum(dim=0).pow(2) + (1 - isTop) * z.sum(dim=0).pow(2)
        balance_loss = config["beta"] * balance_loss.mean()

        return likelihood_loss + quantization_loss + independence_loss + balance_loss

    def update_B_and_H(self, config):
        self.B = (config["alpha"] * self.U + config["gamma"] * self.H).sign()
        self.H = (config["alpha"] * self.Z + config["gamma"] * self.B).sign()

    def calc_loss(self, config):
        s = (self.Y @ self.Y.t() > 0).float()
        inner_product = self.U @ self.Z.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()

        quantization_loss = (self.U - self.B).pow(2) + (self.Z - self.H).pow(2)
        quantization_loss = config["alpha"] * quantization_loss.mean()

        regularization_loss = config["gamma"] * (self.B - self.H).pow(2).mean()

        independence_loss = (self.U.t() @ self.U / self.U.shape[0] - self.I).pow(2) + \
                            (self.Z.t() @ self.Z / self.Z.shape[0] - self.I).pow(2)
        independence_loss = config["lambda"] * independence_loss.mean()

        balance_loss = self.U.sum(dim=0).pow(2) + self.Z.sum(dim=0).pow(2)
        balance_loss = config["beta"] * balance_loss.mean()

        return likelihood_loss + quantization_loss + regularization_loss + independence_loss + balance_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    net_top = config["net"](bit, pretrained=True).to(device)
    net_bottom = config["net"](bit, pretrained=False).to(device)

    optimizer_top = config["optimizer"]["type"](net_top.parameters(), **(config["optimizer"]["optim_params"]))
    optimizer_bottom = config["optimizer"]["type"](net_bottom.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DAPHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net_top.train()
        net_bottom.eval()
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer_top.zero_grad()
            u = net_top(image)
            z = net_bottom(image)
            loss = criterion(u, z, label.float(), ind, config)
            loss.backward()
            optimizer_top.step()

        net_top.eval()
        net_bottom.train()
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer_bottom.zero_grad()
            u = net_top(image)
            z = net_bottom(image)
            loss = criterion(u, z, label.float(), ind, config, isTop=0)
            loss.backward()
            optimizer_bottom.step()

        criterion.update_B_and_H(config)
        train_loss = criterion.calc_loss(config).item()

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net_bottom, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net_bottom, device=device)

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
