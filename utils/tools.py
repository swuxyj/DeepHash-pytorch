from utils.data_list import ImageList
import numpy as np
import random
import torch.utils.data as util_data
from torchvision import transforms
from tqdm import tqdm
import torch
import os
from sklearn.manifold import TSNE
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def image_train(resize_size=256, crop_size=224, dataset="cifar10"):
    transforms_list = [transforms.Resize(resize_size),
                       transforms.CenterCrop(crop_size),
                       transforms.RandomHorizontalFlip()]
    if dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

    elif dataset == "coco":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose(transforms_list +
                              [transforms.ToTensor(),
                               normalize
                               ])


def image_test(resize_size=256, crop_size=224, dataset="cifar10"):
    transforms_list = [transforms.Resize(resize_size),
                       transforms.CenterCrop(crop_size)]
    if dataset == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    elif dataset == "coco":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return transforms.Compose(transforms_list + [
        transforms.ToTensor(),
        normalize
    ])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data(config):
    transform_train = image_train(resize_size=config["resize_size"], crop_size=config["crop_size"],
                                  dataset=config["dataset"])
    transform_test = image_test(resize_size=config["resize_size"], crop_size=config["crop_size"],
                                dataset=config["dataset"])
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    dsets["train_set"] = ImageList(config["data_path"],
                                   open(data_config["train_set"]["list_path"]).readlines(), \
                                   transform=transform_train)
    print("train_set: ", len(dsets["train_set"]))
    dset_loaders["train_set"] = util_data.DataLoader(dsets["train_set"], \
                                                     batch_size=data_config["train_set"]["batch_size"], \
                                                     shuffle=True, num_workers=4)

    dsets["test"] = ImageList(config["data_path"], open(data_config["test"]["list_path"]).readlines(), \
                              transform=transform_test)
    print("test: ", len(dsets["test"]))
    dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                batch_size=data_config["test"]["batch_size"], \
                                                shuffle=False, num_workers=4)

    dsets["database"] = ImageList(config["data_path"], open(data_config["database"]["list_path"]).readlines(), \
                                  transform=transform_test)
    print("database: ", len(dsets["database"]))
    dset_loaders["database"] = util_data.DataLoader(dsets["database"], \
                                                    batch_size=data_config["database"]["batch_size"], \
                                                    shuffle=False, num_workers=4)
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], len(dsets["train_set"]), len(
        dsets["test"])


def compute_result(dataloader, net, usegpu=False):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in dataloader:
        clses.append(cls)
        if usegpu:
            bs.append((net(img.cuda())[0]).data.cpu())
        else:
            bs.append((net(img)[0]).data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)


def visualize(data_loader, net, epoch, GPU, imgDir, logger):
    print("visualize")
    feature_list = []
    label_list = []
    for image, label in tqdm(data_loader):
        if GPU:
            image = image.cuda()
        _, feature = net(image)
        feature_list.append(feature.data.cpu())
        _, label = label.max(1)

        label_list.append(label)
    features = torch.cat(feature_list, 0)
    labels = torch.cat(label_list, 0)

    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
              '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    fig = Figure(figsize=(6, 6), dpi=100)
    fig.clf()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    feat = TSNE(n_components=2).fit_transform(features)

    feat = torch.tensor(feat)
    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], c=colors[i], s=1)
        ax.text(feat[labels == i, 0].mean(), feat[labels == i, 1].mean(), str(i), color='black', fontsize=12)
    ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    canvas.draw()

    if (os.path.exists(imgDir)):
        pass
    else:
        os.makedirs(imgDir)
    fig.savefig(imgDir + '/epoch=%d.jpg' % epoch)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

# https://github.com/jiangqy/DPSH-pytorch/blob/master/utils/CalcHammingRanking.py
def CalcMap(rB, qB, retrievalL, queryL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap
