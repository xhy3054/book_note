#!/usr/bin/env python3
# encoding=utf-8

'''
使用pytorch中的torch.optim包进行参数自动优化的三层全连接神经网络
'''

import torch
from torch.autograd import Variable

batch_n = 100
hidden_layer =100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

#构建网络模型
models = torch.nn.Sequential(
    torch.nn.Linear(input_data, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, output_data)
)

#参数设置
epoch_n = 50
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss()

#构建优化器,传入参数是模型的参数集合,和更新率
optimizer = torch.optim.Adam(models.parameters(), lr = learning_rate)

#训练模型
for epoch in range(epoch_n):
    #获得前向输出,直接将输入数据作为torch.nn.Sequential的输入便可
    y_pred = models(x)
    #计算损失并输出
    loss = loss_fn(y_pred, y)
    print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data))
    #梯度清零
    optimizer.zero_grad()

    #向后传播获得每个节点梯度
    loss.backward()
    #使用优化器自动更新参数
    optimizer.step()


