from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# LCDSH(IJCAI2017)
# paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)
# [LCDSH] epoch:145, bit:48, dataset:cifar10-1, MAP:0.798, Best MAP: 0.798
# [LCDSH] epoch:183, bit:48, dataset:nuswide_21, MAP:0.833, Best MAP: 0.834
def get_config():
    config = {
        "lambda": 3,
        # "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5, "betas": (0.9, 0.999)}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[LCDSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10-1",
        # "dataset": "nuswide_21",
        "epoch": 350,
        "test_map": 5,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class LCDSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(LCDSHLoss, self).__init__()

    def forward(self, u, y, ind, config):
        s = 2 * (y @ y.t() > 0).float() - 1
        inner_product = u @ u.t() * 0.5
        inner_product = inner_product.clamp(min=-50, max=50)
        L1 = torch.log(1 + torch.exp(-s * inner_product)).mean()

        b = u.sign()
        inner_product_ = b @ b.t() * 0.5
        L2 = (inner_product.sigmoid() - inner_product_.sigmoid()).pow(2).mean()

        return L1 + config["lambda"] * L2


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = LCDSHLoss(config, bit)

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
