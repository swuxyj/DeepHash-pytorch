# DeepHash-pytorch
Implementation of Some Deep Hash Algorithms.
#### Any Pull Request is highly welcomed

# How to run
You can easily train and test any algorithm just by
```
pyhon DSH.py
pyhon DPSH.py
pyhon HashNet.py
pyhon DHN.py
```
# Dataset
You can download  ImageNet, NUS-WIDE and COCO dataset [here](https://github.com/thuml/HashNet/tree/master/pytorch),
and download cifar10(Lossless PNG format) [here](https://drive.google.com/open?id=1NZ5QKW2zqzN-RQ4VDpuOAb-UgcsTPUJK)
# Paper And Code
DSH(CVPR2016)  
paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)  
code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)

HashNet(ICCV2017)  
paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)  
code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

DPSH(IJCAI2016)  
paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)   
code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)

DHN(AAAI2016)  
paper [Deep Hashing Network for Efficient Similarity Retrieval](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-hashing-network-aaai16.pdf)  
code [DeepHash-tensorflow](https://github.com/thulab/DeepHash)


# Mean Average Precision,48 bits.
<table>
    <tr>
        <td>Algorithms</td><td>dataset</td><td>this impl.</td><td>paper</td>
    </tr>
    <tr>
        <td >DSH</td><td >cifar10</td> <td >0.787</td> <td >0.6755</td>
    </tr>
    <tr>
        <td ></td><td >nus wide21</td> <td >0.7510</td> <td >0.5621</td>
    </tr>
    <tr>
        <td >DPSH</td><td >cifar10</td> <td >0.775</td> <td >0.757</td>
    </tr>
    <tr>
        <td ></td><td >nus wide21</td> <td >0.814</td> <td >0.851</td>
    </tr>
    <tr>
        <td >HashNet</td><td >nus wide81</td> <td >0.764</td> <td >0.7114</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.600</td> <td >0.6633</td>
    </tr>
    <tr>
        <td >DHN</td><td >cifar10</td> <td >0.779</td> <td >0.621</td>
    </tr>
    <tr>
        <td ></td><td >nus wide21</td> <td >0.816</td> <td >0.758</td>
    </tr>
</table>
