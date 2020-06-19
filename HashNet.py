from utils.tools import *
from network import *
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import time

plt.switch_backend('agg')
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        # "optimizer":{"type":  optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.SGD,
                      "optim_params": {"lr": 0.0003, "weight_decay": 10 ** -5, "momentum": 0.9, "nesterov": True},
                      "lr_type": "step"},
        "info": "[HashNet]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "nuswide_21",
        # "dataset":"coco",
        # "dataset":"nuswide_81",
        # "dataset":"imagenet",

        "epoch": 5000,
        "test_map": 10,
        "GPU": True,
        # "GPU":False,
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
        config["n_class"] = 91
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


# from https://github.com/thuml/HashNet/issues/17
def pairwise_loss(outputs1, outputs2, label1, label2, config):
    similarity = (label1 @ label2.t() > 0).float()
    dot_product = config["alpha"] * outputs1 @ outputs2.t()

    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    if config["GPU"]:
        exp_loss = torch.log(1 + torch.exp(-torch.abs(dot_product))) \
                   + torch.max(dot_product, torch.FloatTensor([0.]).cuda()) - similarity * dot_product
    else:
        exp_loss = torch.log(1 + torch.exp(-torch.abs(dot_product))) \
                   + torch.max(dot_product, torch.FloatTensor([0.])) - similarity * dot_product

    # weight
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0 + S1
    exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
    exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

    loss = torch.sum(exp_loss) / S

    return loss


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    net = config["net"](bit, "HashNet")
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    U = torch.zeros(num_train, bit).float()
    L = torch.zeros(num_train, config["n_class"]).float()
    if config["GPU"]:
        U = U.cuda()
        L = L.cuda()

    for epoch in range(config["epoch"]):
        net.scale = (epoch // config["step_continuation"] + 1) ** 0.5
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], net.scale), end="")

        net.train()
        train_loss = 0

        for image, label, ind in train_loader:
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()
            b = net(image)

            U[ind, :] = b.data
            L[ind, :] = label.float()

            loss = pairwise_loss(b, U, label.float(), L, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

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
        config["alpha"] = 10. / bit
        train_val(config, bit)
