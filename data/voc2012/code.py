import os
import numpy as np
# all_label_data = np.zeros((17125,20),dtype=np.int8)
all_label_data = np.zeros((11540,20),dtype=np.int8)
label_dir_name = "VOCdevkit\VOC2012\ImageSets\Main"
label2index = {}

image2index = {}
index2image = {}
for label_file in os.listdir(label_dir_name):
    if len(label_file.split("_")) == 2:
        label, _ = label_file.split("_")
        if label in label2index:
            label_index = label2index[label]
        else:
            label_index = len(label2index)
            label2index[label] = label_index
        with open(os.path.join(label_dir_name,label_file), "r") as f:
            for line in f.readlines():
                if line.split(" ")[-1].strip() == "1":
                    image_name = line.split(" ")[0]
                    if image_name in image2index:
                        image_index = image2index[image_name]
                    else:
                        image_index = len(image2index)
                        index2image[image_index] = image_name
                        image2index[image_name] = image_index
                    all_label_data[image_index][label_index] = 1

train_num = 4000
test_num = 1000
perm_index = np.random.permutation(all_label_data.shape[0])

train_data_index = perm_index[:train_num]
test_data_index = perm_index[train_num:train_num+test_num]
database_data_index = perm_index[train_num+test_num:]

with open("database.txt", "w") as f:
    for index in database_data_index:
        line = "VOCdevkit/VOC2012/JPEGImages/" + index2image[index] + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", "," ") + "\n"
        f.write(line)
with open("train.txt", "w") as f:
    for index in train_data_index:
        line = "VOCdevkit/VOC2012/JPEGImages/" + index2image[index] + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", "," ") + "\n"
        f.write(line)
with open("test.txt", "w") as f:
    for index in test_data_index:
        line = "VOCdevkit/VOC2012/JPEGImages/" + index2image[index] + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", "," ") + "\n"
        f.write(line)
