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
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[0.1]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        # "dataset": "nuswide_21",
        # "dataset":"coco",
        "dataset":"nuswide_81",
        # "dataset":"imagenet",
        "featureImg": "SNE",
        # "featureImg":"PCA",
        "epoch": 500,
        "test_map": 10,
        "GPU": True,
        # "GPU":False,
        "bit_list": [48],
    }
    if config["dataset"] == "cifar10":
        config["topK"] = -1
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


def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.8 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def calc_loss(x1, x2, y1, y2, config):
    dist = (x1.unsqueeze(1) - x2.unsqueeze(0)).pow(2).sum(dim=2)
    y = (y1 @ y2.t() == 0).float()

    loss = (1 - y) / 2 * dist + y / 2 * (config["m"] - dist).clamp(min=0)
    loss1 = loss.mean()
    loss2 = config["alpha"] * (1 - x1.sign()).abs().mean()
    # tqdm.write("%.3f,%.3f"%(loss1.item(),loss2.item()))
    return loss1 + loss2


def train_val(config, bit):
    config["m"] = 2 * bit
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    net = config["net"](bit)
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    U = torch.zeros(num_train, bit).float()
    L = torch.zeros(num_train, config["n_class"]).float()

    if config["GPU"]:
        U = U.cuda()
        L = L.cuda()

    for epoch in range(config["epoch"]):
        # if config["dataset"] == "cifar10":
        #     net.eval()
        #     visualize(test_loader, net, epoch,config["GPU"],config["featureImg"],logger)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        optimizer = AdjustLearningRate(optimizer, epoch, config["optimizer"]["optim_params"]["lr"])
        train_loss = 0
        for image, label, ind in train_loader:
            if config["GPU"]:
                image, label = image.cuda(), label.cuda()

            optimizer.zero_grad()
            b, _ = net(image)
            for i, ind in enumerate(ind):
                U[ind, :] = b.data[i]
                L[ind, :] = label[i]

            loss = calc_loss(b, U, label.float(), L, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            # print("calculating dataset binary code.......")
            # trn_binary, trn_label = compute_result(train_loader, net, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

            # print("calculating map.......")
            if config["topK"] > 0:
                mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                 config["topK"])
            else:
                mAP = CalcMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())

            print(
                "%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
