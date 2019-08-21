from utils.tools import *
from models.dpsh import *

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import time

plt.switch_backend('agg')
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "alpha": 0.1,
        "nu": 0.1,
        "mu": 1,
        "eta": 55,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[mean]",
        "resize_size": 256,
        "crop_size": 224,
        "dcc_iter": 10,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset":"nuswide_21",
        # "dataset":"coco",
        # "dataset":"nuswide_81",
        # "dataset":"imagenet",
        "epoch": 500,
        "test_map": 10,
        "GPU": True,
        # "GPU":False,
        "device": torch.device("cuda:0"),
        # "device": torch.device("cpu"),
        "bit_list": [48],
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
    config["data_path"] = "../dataset/" + config["dataset"] + "/"
    if config["dataset"][:7] == "nuswide":
        config["data_path"] = "../dataset/nus_wide/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config




def calc_loss(S, outputs, U, B, W, L, config, index):
    inner_product = outputs @ U * 0.5

    if config["GPU"]:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]).cuda())
    else:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]))
    loss = log_trick - S * inner_product
    loss = loss.mean()

    # Classification loss
    cl_loss = (L[:, index] - W.t() @ B[:, index]).pow(2).mean()

    # Regularization loss
    reg_loss = W.pow(2).mean()

    loss = loss + config["mu"] * cl_loss + config["nu"] * reg_loss
    return loss


def solve_dcc(W, Y, H, B, eta, mu):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem
    """
    for i in range(B.shape[0]):
        P = W @ Y + eta / mu * H

        p = P[i, :]
        w = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i + 1:, :]))
        B_prime = torch.cat((B[:i, :], B[i + 1:, :]))

        B[i, :] = (p - B_prime.t() @ W_prime @ w).sign()

    return B


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    net = config["net"](bit)
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    U = torch.zeros(num_train, bit).float().t()
    B = torch.zeros(num_train, bit).float().t()
    L = torch.zeros(num_train, config["n_class"]).float().t()

    for image, label, ind in train_loader:
        L[:,ind] = label.float().t()

    if config["GPU"]:
        U = U.cuda()
        B = B.cuda()
        L = L.cuda()
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            label = label.float()
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()

            outputs = net(image)

            U[:, ind] = outputs.data.t()

            S = (label @ L > 0).float()

            for dit in range(config["dcc_iter"]):
                # W-step
                if config["GPU"]:
                    W = torch.inverse(B @ B.t() + config["nu"] / config["mu"] * torch.eye(bit).cuda()) @ B @ L.t()
                else:
                    W = torch.inverse(B @ B.t() + config["nu"] / config["mu"] * torch.eye(bit)) @ B @ L.t()

                # B-step
                B = solve_dcc(W, L, U, B, config["eta"], config["mu"])

            loss = calc_loss(S, outputs, U, B, W, L, config, ind)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            print("calculating dataset binary code.......")
            # trn_binary, trn_label = compute_result(train_loader, net, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

            print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            print(
                "%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
