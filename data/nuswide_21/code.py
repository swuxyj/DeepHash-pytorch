# After download  NUS-WIDE.tar.gz(https://github.com/TreezzZ/DSDH_PyTorch)
# you can use code.py to  to generate database.txt 、 train.txt and test.txt of nuswide_21

import numpy as np
database_img = []
with open("database_img.txt") as f:
    for line in f.readlines():
        database_img.append(line.strip())
test_img = []
with open("test_img.txt") as f:
    for line in f.readlines():
        test_img.append(line.strip())
test_label_onehot = []
with open("test_label_onehot.txt") as f:
    for line in f.readlines():
        test_label_onehot.append(line.strip())
database_label_onehot = []
with open("database_label_onehot.txt") as f:
    for line in f.readlines():
        database_label_onehot.append(line.strip())
test_label = []
with open("test_label.txt") as f:
    for line in f.readlines():
        test_label.append(line.strip())
database_label = []
with open("database_label.txt") as f:
    for line in f.readlines():
        database_label.append(line.strip())

database_img.extend(test_img)
database_label.extend(test_label)
database_label_onehot.extend(test_label_onehot)

train_num = 500
test_num = 100
train_count = [train_num] * 21
test_count = [test_num] * 21
train_data = []
test_data = []
database_data = []

import numpy as np

perm_index = np.random.permutation(len(database_img))

for index in perm_index:
    line = database_img[index] + " " + database_label_onehot[index] + "\n"
    add_position = "database"
    for classnum in database_label[index].split():
        classnum = int(classnum)
        if train_count[classnum]:
            add_position = "train"
            train_count[classnum] -= 1
            break
        if test_count[classnum]:
            add_position = "test"
            test_count[classnum] -= 1
            break

    if add_position == "train":
        train_data.append(line)
        database_data.append(line)
    elif add_position == "test":
        test_data.append(line)
    else:
        database_data.append(line)

with open("database.txt", "w") as f:
    for line in database_data:
        f.write(line)
with open("train.txt", "w") as f:
    for line in train_data:
        f.write(line)
with open("test.txt", "w") as f:
    for line in test_data:
        f.write(line)
