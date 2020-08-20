from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# to do
# ADSH(AAAI2018)
# paper [Asymmetric Deep Supervised Hashing](https://cs.nju.edu.cn/lwj/paper/AAAI18_ADSH.pdf)
# code1 [ADSH matlab + pytorch](https://github.com/jiangqy/ADSH-AAAI2018)
# code2 [ADSH_pytorch](https://github.com/TreezzZ/ADSH_PyTorch)

