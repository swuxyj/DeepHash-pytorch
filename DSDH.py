from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')


# DSDH(NIPS2017)
# paper [Deep Supervised Discrete Hashing](https://papers.nips.cc/paper/6842-deep-supervised-discrete-hashing.pdf)
# code [DSDH_PyTorch](https://github.com/TreezzZ/DSDH_PyTorch)

def get_config():
    config = {
        "alpha": 1,
        "nu": 1,
        "mu": 1,
        "eta": 55,
        "dcc_iter": 10,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[DSDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        # "dataset": "cifar10-1",
        "dataset": "cifar10-2",
        # "dataset": "mirflickr",
        # "dataset": "voc2012
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        # "dataset": "coco",
        # "dataset":"imagenet",
        "epoch": 150,
        "test_map": 15,
        "save_path": "save/DSDH",
        "GPU": True,
        # "GPU":False,
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DSDHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSDHLoss, self).__init__()
        self.U = torch.zeros(bit, config["num_train"]).float()
        self.B = torch.zeros(bit, config["num_train"]).float()
        self.Y = torch.zeros(config["n_class"], config["num_train"]).float()

        if config["GPU"]:
            self.U = self.U.cuda()
            self.B = self.B.cuda()
            self.Y = self.Y.cuda()

    def forward(self, u, y, ind, config):

        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        self.updateBandW()

        inner_product = u @ self.U * 0.5
        s = (y @ self.Y > 0).float()


        likelihood_loss = (1 + (-inner_product.abs()).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        # Classification loss
        cl_loss = (y.t() - self.W.t() @ self.B[:, ind]).pow(2).mean()

        # Regularization loss
        reg_loss = self.W.pow(2).mean()

        loss = likelihood_loss + config["mu"] * cl_loss + config["nu"] * reg_loss
        return loss

    def updateBandW(self):
        B = self.B
        for dit in range(config["dcc_iter"]):
            # W-step
            if config["GPU"]:
                W = torch.inverse(B @ B.t() + config["nu"] / config["mu"] * torch.eye(bit).cuda()) @ B @ self.Y.t()
            else:
                W = torch.inverse(B @ B.t() + config["nu"] / config["mu"] * torch.eye(bit)) @ B @ self.Y.t()
            for i in range(B.shape[0]):
                P = W @ self.Y + config["eta"] / config["mu"] * self.U
                p = P[i, :]
                w = W[i, :]
                W_prime = torch.cat((W[:i, :], W[i + 1:, :]))
                B_prime = torch.cat((B[:i, :], B[i + 1:, :]))
                B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

        self.B = B
        self.W = W


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit)
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DSDHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

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
            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

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
