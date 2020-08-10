from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')


# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)

def get_config():
    config = {
        "alpha": 0.5,
        "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 60,
                      "optim_params": {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}},

        # "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[GreedyHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet,
        "dataset": "cifar10",
        # "dataset": "coco",
        # "dataset": "imagenet",
        "epoch": 200,
        "test_map": 15,
        "save_path": "save/GreedyHash",
        "GPU": True,
        # "GPU":False,
        "bit_list": [48],
    }
    config = config_dataset(config)
    return config


class GreedyHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        if config["GPU"]:
            self.fc = self.fc.cuda()
            self.criterion = self.criterion.cuda()

    def forward(self, u, onehot_y, ind, config):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.topk(1)[1].squeeze(1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        loss2 = config["alpha"] * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit)
    if config["GPU"]:
        net = net.cuda()

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = GreedyHashLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.6f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

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
