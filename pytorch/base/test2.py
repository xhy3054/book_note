#!/usr/bin/env python3
# encoding:utf-8

'''
使用torch.autograd.Variable类作为节点来构造一个可以自动进行后向传播求导的三层神经网络
'''
import torch
# 导入自动梯度包中的Variable模块
from torch.autograd import Variable

# 每批次输入数据的数量
batch_n = 100
# 经过隐藏层后保留的数据特征的个数
hidden_layer = 100
# 每个输入数据包含的数据特征个数
input_data = 1000
# 输出的数据的范围
output_data = 10

# 将运算中整个网络参数都封装成 torch.autograd 类对象
# x为输入，y为输出节点
x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

# w1与w2为连接三层网络的两个参数节点，这也是需要进行自动梯度调整的地方
w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)
 
# 迭代次数
epoch_n = 20
# 学习速率，也就是每次微调的步长
learning_rate = 1e-6

for epoch in range(epoch_n):
    #直接将整个前向运算整个写出来    
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y).pow(2).sum()
    print("Epoch:{}, Loss:{:.4f}".format(epoch,loss.data))

    #自动进行后向传播
    loss.backward()

    #后向传播后，每个参数节点中会存放梯度数据
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data

    #更新后，将各个参数节点的梯度值全部置零
    w1.grad.data.zero_()
    w2.grad.data.zero_()


