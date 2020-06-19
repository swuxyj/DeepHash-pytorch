from utils.tools import *
from network import *

import torch
import torch.optim as optim
import time

torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "alpha": 0.05,
        "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}, "lr_type": "step"},
        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[DPSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        # "net": AlexNet,
        "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "nuswide_21",
        # "dataset":"coco",
        # "dataset":"nuswide_81",
        # "dataset":"imagenet",
        "epoch": 90,
        "evaluate_freq": 15,
        "GPU": True,
        # "GPU":False,
        "bit_list": [48, 32, 24, 12],
    }
    if config["dataset"] == "cifar10":
        config["topK"] = 54000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"][:7] == "nuswide":
        config["data_path"] = "/dataset/nus_wide/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


def calc_loss(x1, x2, y1, y2, config):
    s = (y1 @ y2.t() > 0).float()
    inner_product = x1 @ x2.t() * 0.5
    if config["GPU"]:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]).cuda())
    else:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]))
    loss = log_trick - s * inner_product
    loss1 = loss.mean()
    loss2 = config["alpha"] * (x1 - x1.sign()).pow(2).mean()
    return loss1 + loss2


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    net = config["net"](bit)
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    U = torch.zeros(num_train, bit).float()
    L = torch.zeros(num_train, config["n_class"]).float()

    if config["GPU"]:
        U = U.cuda()
        L = L.cuda()
    Best_mAP = 0
    for epoch in range(config["epoch"]):
        scheduler.step()

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, index in train_loader:
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()
            b = net(image)

            U[index, :] = b.data
            L[index, :] = label.float()

            loss = calc_loss(b, U, label.float(), L, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["evaluate_freq"] == 0:

            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            # trn_binary, trn_label = compute_result(train_loader, net, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),config["topK"])

            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            if mAP > Best_mAP:
                Best_mAP = mAP
    print("bit:%d,Best MAP:%.3f" % (bit, Best_mAP))


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
