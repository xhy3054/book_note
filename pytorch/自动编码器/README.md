# 自动编码器
一般自动编码器主要由两部分组成：编码器与解码器
    - 编码器负责从原始数据中提取出最有用的特征，丢弃不是很有用的信息
    - 解码器负责利用编码器提取出来的特征恢复出原始数据

作用：**提纯数据**，因为原始数据中的噪声与不重要的特性在编码过程中被丢弃了，所以解码出来的往往是最纯粹的数据。

## pytorch实战
目的：祛除图像噪声

1. 构造线性自动编码器，通过训练得到可以祛除噪声的网络。网络结构如下
    - 线性层（28*28 -> 128）
    - ReLU
    - 线性层（128 -> 64）
    - ReLU
    - 线性层（64 -> 32）
    - ReLU

    - 线性层（32 -> 64）
    - ReLU
    - 线性层（64 -> 128）
    - ReLU
    - 线性层（128 -> 28*28）

2. 构造卷积自动编码器，通过训练得到可以祛除噪声的网络
    - 卷积层（1,64，kernel_size=3, stride=1, padding=1）
    - ReLU
    - MaxPool2d(kernel_size=2, stride=2)
    - 卷积层（64,128，kernel_size=3, stride=1, padding=1）
    - ReLU
    - MaxPool2d(kernel_size=2, stride=2)


    - Upsample(scale_factor=2, mode="nearest")
    - 卷积层（128,64，kernel_size=3, stride=1, padding=1）
    - ReLU
    - Upsample(scale_factor=2, mode="nearest")
    - 卷积层（64,1，kernel_size=3, stride=1, padding=1）

