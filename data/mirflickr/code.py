# download data from here: https://press.liacs.nl/mirflickr/mirdownload.html

# import hashlib
# with open("mirflickr25k.zip","rb") as f:
#     md5_obj = hashlib.md5()
#     md5_obj.update(f.read())
#     hash_code = md5_obj.hexdigest()
#     print(str(hash_code).upper() == "A23D0A8564EE84CDA5622A6C2F947785")

import os
import numpy as np

all_label_data = np.zeros((25000, 38), dtype=np.int8)
label_index = -1
label_dir_name = "mirflickr25k_annotations_v080"
for label_file in os.listdir(label_dir_name):
    if "README.txt" != label_file:
        label_index += 1
        with open(os.path.join(label_dir_name, label_file), "r") as f:
            for line in f.readlines():
                all_label_data[int(line.strip()) - 1][label_index] = 1

train_num = 4000
test_num = 1000
perm_index = np.random.permutation(all_label_data.shape[0])

train_data_index = perm_index[:train_num]
test_data_index = perm_index[train_num:train_num + test_num]
database_data_index = perm_index[train_num + test_num:]

with open("database.txt", "w") as f:
    for index in database_data_index:
        line = "im" + str(index + 1) + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", ", " ") + "\n"
        f.write(line)
with open("train.txt", "w") as f:
    for index in train_data_index:
        line = "im" + str(index + 1) + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", ", " ") + "\n"
        f.write(line)
with open("test.txt", "w") as f:
    for index in test_data_index:
        line = "im" + str(index + 1) + ".jpg " + str(all_label_data[index].tolist())[1:-1].replace(", ", " ") + "\n"
        f.write(line)
