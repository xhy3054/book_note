#!/usr/bin/env python3
# encoding=utf-8

'''
使用pytorch中的torch.nn包进行神经网络模型的搭建与模块化的损失函数
'''

import torch
from torch.autograd import Variable
#必要尺度参数还是得有的
batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10
#输入输出尺寸也是固定的
x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

#下面是整个三层网络的参数与计算模型，无需使用矩阵乘法自行构建了
#使用torch.nn直接构造，由 全连接+激活函数+全连接层 三部分计算组成
'''
models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data)
)
'''
#另一种模型的构建模式，使用orderdict有序字典进行
#import collections.OrderedDict as od
from collections import OrderedDict
models = torch.nn.Sequential(OrderedDict([
    ("Line1", torch.nn.Linear(input_data, hidden_layer)),
    ("Relu1", torch.nn.ReLU()),
    ("Line2", torch.nn.Linear(hidden_layer, output_data))])    
)


#查看模型结构
print(models)

#对搭建好的模型进行参数优化
epoch_n = 10000
learning_rate = 1e-4
#损失函数设置为torch.nn提供的MSELoss来计算损失值（均方误差）
loss_fn = torch.nn.MSELoss()

for epoch in range(epoch_n):
    #为模型传入输入，得到前向输出
    y_pred = models(x)
    #使用定义好的损失函数计算损失
    loss=loss_fn(y_pred, y)

    if epoch%1000 == 0:
        print(loss)
        print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data))
    #清零，因为下面马上就进行后向传播了
    models.zero_grad()

    #后向传播
    loss.backward()

    #参数调整    
    for param in models.parameters():
        param.data -= param.grad.data*learning_rate


#

