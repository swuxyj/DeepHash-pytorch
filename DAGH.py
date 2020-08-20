from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# to do
# DAGH(ICCV2019)
# paper[Deep Supervised Hashing with Anchor Graph](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Deep_Supervised_Hashing_With_Anchor_Graph_ICCV_2019_paper.pdf)
# code [DAGH-Matlab](http://www.scholat.com/personalPaperList.html?Entry=laizhihui&selectType=allPaper)

