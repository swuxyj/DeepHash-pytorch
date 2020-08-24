from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# DFH(BMVC2019)
# paper [Push for Quantization: Deep Fisher Hashing](https://arxiv.org/abs/1909.00206)
# code [Push-for-Quantization-Deep-Fisher-Hashing](https://github.com/liyunqianggyn/Push-for-Quantization-Deep-Fisher-Hashing)

def get_config():
    config = {
        "m": 3,
        "mu": 0.1,
        "vul": 1,
        "nta": 1,
        "eta": 0.5,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DFH]",
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
        "epoch": 150,
        "test_map": 5,
        # "save_path": "save/DFH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class DFHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DFHLoss, self).__init__()
        self.U = torch.zeros(bit, config["num_train"]).float().to(config["device"])
        self.Y = torch.zeros(config["n_class"], config["num_train"]).float().to(config["device"])

        # Relax_center
        self.V = torch.zeros(bit, config["n_class"]).to(config["device"])

        # Center
        self.C = self.V.sign().to(config["device"])

        T = 2 * torch.eye(self.Y.size(0)) - torch.ones(self.Y.size(0))
        TK = self.V.size(0) * T
        self.TK = torch.FloatTensor(torch.autograd.Variable(TK, requires_grad=False)).to(config["device"])


    def forward(self, u, y, ind, config):

        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        b = (config["mu"] * self.C @ y.t() + u.t()).sign()

        self.Center_gradient(torch.autograd.Variable(self.V, requires_grad=True),
                             torch.autograd.Variable(y, requires_grad=False),
                             torch.autograd.Variable(b, requires_grad=False))

        s = (y @ self.Y > 0).float()
        inner_product = u @ self.U * 0.5
        inner_product = inner_product.clamp(min=-100, max=50)
        metric_loss = ((1 - s) * torch.log(1 + torch.exp(config["m"] + inner_product))
                       + s * torch.log(1 + torch.exp(config["m"] - inner_product))).mean()
        # metric_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()  # Without Margin
        quantization_loss = (b - u.t()).pow(2).mean()
        loss = metric_loss + config["eta"] * quantization_loss
        return loss

    def Center_gradient(self, V, batchy, batchb):
        alpha = 0.03
        for i in range(200):
            intra_loss = (V @ batchy.t() - batchb).pow(2).mean()
            inter_loss = (V.t() @ V - self.TK).pow(2).mean()
            quantization_loss = (V - V.sign()).pow(2).mean()

            loss = intra_loss + config["vul"] * inter_loss + config["nta"] * quantization_loss

            loss.backward()

            if i in (149, 179):
                alpha = alpha * 0.1

            V.data = V.data - alpha * V.grad.data

            V.grad.data.zero_()
        self.V = V
        self.C = self.V.sign()


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DFHLoss(config, bit)

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
