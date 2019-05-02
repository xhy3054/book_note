#!/usr/bin/env python3
# encoding=utf-8

'''
手写数字识别,pytorch框架,mnist数据集
'''

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
#%matplotlib inline

#数据预处理,此处需要加上一个自定义的lambda转换,用于将单通道转成三通道,或者将转换操作变成对单通道图像的操作

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean = [0.5,0.5,0.5],std=[0.5,0.5,0.5])])
'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5,],std=[0.5,])])
'''
#下载数据集,其中
#root是数据集下载后的存放路径,
#transform指定导入数据集时进行的变换操作
#train指定在数据集下载完成后需要载入数据集的哪部分数据,true则载入该数据集的训练集部分
data_train = datasets.MNIST(root = "../../data/",
                            transform = transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root = "../../data/",
                        transform = transform,
                        train = False)

#数据装载
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True,
                                                )
#batch_size确认每个包的大小
#shuffle确认是否在装载过程中打乱图片顺序
data_loader_test = torch.utils.data.DataLoader( dataset=data_test,
                                                batch_size=64,
                                                shuffle=True,
                                                )

print(len(data_train))

'''
#选取其中一个批次的数据进行预览
images, labels = next(iter(data_loader_train))

#此处make_grid函数的参数是载入的图像数据，装载数据是4维的，维度从前往后分别是batch_size,channel,height,weight。
#make_grid函数的作用是将上述装载数据编程三维，其中第一维度batch_size被平铺到一个维度，所以现在返回的img是三维的，（channe,height,weight）
img = torchvision.utils.make_grid(images)

#如果想要使用plt显示出正常的图片形式，我们需要首先的到它的数组形式，然后使用transpose将色彩维度channel挪到最后得到正常的图形形式（其实就是本来正面是夹层，如今我们给他反了个身）
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]

#std = [0.5]
#mean = [0.5]

img = img*std + mean
print([labels[i] for i in range(64)])
#plt.imshow(img)
#plt.show()
'''
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2,kernel_size=2))
    
        self.dense=torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10))

    def forward(self, x):
        x = self.conv1(x)
        #此处对参数进行扁平化处理,将原本3维变成1维
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x

#实例化模型,损失函数使用交叉熵,优化函数使用Adam自适应优化算法
model = Model()
print(model)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)

    for data in data_loader_train:
        #此处应该是data分两部分,数据与真值
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        #前向
        outputs = model(X_train)
        #此处torch.ax(data,1)返回一个tuple元组，第一个元素为每一行最大值的那个元素，第二个元素为最大元素的列索引
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        #计算损失
        loss = cost(outputs, y_train)
        #后向并自动优化
        loss.backward()
        optimizer.step()


        running_loss += loss.data
        running_correct += torch.sum(pred==y_train.data)

    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _,pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train), 100*running_correct/len(data_train), 100*testing_correct/len(data_test))) 
