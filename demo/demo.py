import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


data_dir = "/data/xyj/2021/DeepHash-pytorch/save/DPSH/imagenet_64bits_0.8824931967229359/"
trn_binary = np.load(data_dir + "trn_binary.npy")
trn_label = np.load(data_dir + "trn_label.npy")
tst_binary = np.load(data_dir + "tst_binary.npy")
tst_label = np.load(data_dir + "tst_label.npy")

img_dir = "/dataset/imagenet/"
with open("./../data/imagenet/database.txt", "r") as f:
    trn_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]
with open("./../data/imagenet/test.txt", "r") as f:
    tst_img_path = [img_dir + item.split(" ")[0] for item in f.readlines()]

m = 5
n = 8
plt.figure(figsize=(40, 20), dpi=50)
font_size = 30
tst_select_index = np.random.permutation(range(tst_binary.shape[0]))[0: m]
for row, query_index in enumerate(tst_select_index):
    query_binary = tst_binary[query_index]
    query_label = tst_label[query_index]
    # 计算测试集和检索是否相似
    gnd = (np.dot(query_label, trn_label.transpose()) > 0).astype(np.float32)
    # 通过哈希码计算汉明距离
    hamm = CalcHammingDist(query_binary, trn_binary)
    # 计算最近的n个距离的索引
    ind = np.argsort(hamm)[:n]
    # 返回结果的真值
    t_gnd = gnd[ind]
    # 返回结果的汉明距离
    q_hamm = hamm[ind].astype(int)

    q_img_path = tst_img_path[query_index]
    return_img_list = np.array(trn_img_path)[ind].tolist()

    plt.subplot(m, n + 1, row * (n+1) + 1)

    img = Image.open(q_img_path).convert('RGB').resize((128, 128))
    plt.imshow(img)
    plt.axis('off')
    plt.text(5, 145, 'query image', size=font_size)

    for index, img_path in enumerate(return_img_list):
        # plt.subplot(1, n + 1, index + 2)
        plt.subplot(m, n + 1, row * (n+1) + index + 2)
        img = Image.open(img_path).convert('RGB').resize((120, 120))
        if t_gnd[index]:
            plt.text(60, 145, '√', size=font_size)
            img = ImageOps.expand(img, 4, fill=(0, 0, 255))
        else:
            plt.text(60, 145, '×', size=font_size)
            img = ImageOps.expand(img, 4, fill=(255, 0, 0))
        plt.axis('off')
        plt.imshow(img)
plt.savefig("demo.png")
plt.show()
