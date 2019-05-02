## `torch.nn.Module`
`torch.nn.Module`类是一个抽象类，代表一个神经网络模型，一般自定义的网络模型都会继承这个类，并且重写以下函数
    - `__init__`: 这个函数主要是复杂网络的模型的初始化
    - `forward`: 重写了这个函数后就可以使用`model(input)`来完成网络的前向计算了

## 使用pytorch自定义传播函数
> 代码请见同目录下[test3](test3.py)

```python
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
```
---
如上，继承自`torch.nn.Module`的类Model是用来定义前向传播函数与后向传播函数的。

```python
model = Model()
x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
y = Variable(torch.randn(batch_n, output_data), requires_grad = False)

w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)
# 唯一不同的地方，第一前向传播的方式
y_pred = model(x, w1, w2)
```
---
如上，便定义了模型的前向传播方式
