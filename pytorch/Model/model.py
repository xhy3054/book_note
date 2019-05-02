#!/usr/bin/env python3
# encoding: utf-8
'''
自定义前向传播函数与后向传播函数，来优化一个简易的三层神经网络
'''
import torch
from torch.autograd import Variable

batch_n = 64
hidden_layer = 100
input_data = 1000
output_data = 10

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, w1, w2):
        # 定义前向传播中的矩阵运算
        x = torch.mm(input, w1)
        x = torch.clamp(x, min = 0)
        x = torch.mm(x, w2)
        return x

    def backward(self):
        # 没有特殊需求，后向传播一般无需修改
        pass

if __name__ == '__main__':
    model = Model()
    
    x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
    y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

    w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
    w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)

    epoch_n =30
    learning_rate = 1e-6

    for epoch in range(epoch_n):
        # 唯一不同的地方，第一前向传播的方式
        y_pred = model(x, w1, w2)
        
        loss = (y_pred - y).pow(2).sum()
        print("Epoch:{}, Loss:{:.4f}".format(epoch, loss.data))
        loss.backward()

        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
