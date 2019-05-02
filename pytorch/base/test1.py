#!/usr/bin/env python3
# encoding:utf-8

'''
本例程使用pytorch实现一个传统的三层全连接神经网络
'''

import torch
# 每批次输入数据的数量
batch_n = 100
# 经过隐藏层后保留的数据特征的个数
hidden_layer = 100
# 输入数据包含的数据特征
input_data = 1000
# 输出的数据的范围
output_data = 10

'''
完整流程：先输入100个具有1000个特征的数据，经过隐藏层后变成100个具有100个特征的数据，再经过输出层后输出100个具有10个分类结果值的数据，在得到输出结果后计算损失并进行后向传播。这样一次模型的训练就完成了，然后循环这个流程完成指定次数的迭代，达到优化模型参数的目的。
'''
# 输入层输入数据初始化，维度（100,1000）
x = torch.randn(batch_n, input_data)
# 输出层真值初始化，维度（100,10）
y = torch.randn(batch_n, output_data)

# 输入层到隐藏层的权重参数维度（1000，100）
w1 = torch.randn(input_data, hidden_layer)
# 隐藏层到输出蹭的权重参数维度（100,10）
w2 = torch.randn(hidden_layer, output_data)

# 迭代次数
epoch_n = 20
# 学习速率，也就是每次微调的步长
learning_rate = 1e-6

# 开始
for epoch in range(epoch_n):
    # 输入层到隐藏层的矩阵计算，此处h1是隐藏层输出的结果矩阵
    h1 = x.mm(w1)
    # 非线性的激励层，此处使用relu函数
    h1 = h1.clamp(min=0)
    # 隐藏层到输出层的矩阵计算
    y_pred = h1.mm(w2)
# 到这里前向传播完成    

    # 计算损失函数 均方误差
    loss = (y_pred - y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss))

# 接下来进行后向传播
    # 计算输出的误差梯度
    grad_y_pred = 2*(y_pred-y)
    # 得到w2的误差梯度矩阵
    grad_w2 = h1.t().mm(grad_y_pred)

    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp(min=0)
    # 得到w1的误差梯度矩阵
    grad_w1 = x.t().mm(grad_h)

    # 参数矩阵的更新
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2

    
