# 装载数据
## Dataset类
`torch.utils.data.Dataset`是一个代表一个数据集的抽象类。你定制的数据集应该集成这个抽象类，并重写下面的方法
    - `__len__`: 重写了之后就可以调用`len(dataset)`返回数据集的大小了
    - `__getitem__`: 重写之后你可以使用`dataset[i]`来索引某一个样本实例了

### 实例演示
> 此处使用pytorch[官网](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)上的教程示例代码。

原始数据集描述：我们下载了一个[人脸数据集](https://download.pytorch.org/tutorial/faces.zip)，这个数据集主要分为两部分，第一部分是人脸图片，第二部分是一个`csv`文件，记录了每张人脸上68个关键点的坐标

数据集类创建：
    - 首先，我们在初始化函数`__init__`中读入csv文件，并初始化数据集地址与预处理操作(transform)
    - 在函数`__getitem__`中读入图片，这是因为将所有图片都存入内存是不现实的，所以我们只在需要时进行读入
    - 在函数`__getitem__`中还需要进行将返回的数据组织成一个字典形式`{'image': image, 'landmarks': landmarks}`
    - 在函数`__getitem__`中还需要进行transform操作（如果定义了的话）
    - 定义如下

```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```
---

## 数据集预处理
很多数据集原始的数据或多或少存在一些问题使得不能直接将其输入网络，比如上面的数据集中图片的尺寸大小就是不一致的，大多数神经网络需要它们具有一致的尺寸。这个时候在输入网络之前我们需要对原始数据集进行预处理transform操作。下面我们会进行如下三个预处理操作：
    - `Rescale`: 对图片进行尺度缩放
    - `RandomCrop`: 
    - `ToTensor`: 将原本numpy矩阵形式的图片转换成torch的tensor矩阵形式的图片（此处会交换轴由(012)变成(201)）

此处我们会为上述每一个操作编写一个类，只要在初始化对象时确定参数，就可以一直使用这个对象对数据集样本进行相应的变换操作。每个变换操作类会实现两个方法
    - `__init__`: 该类对象的初始化操作
    - `__call__`: 在进行变换操作`tran(sample)`时执行的函数

### `torchvision.transforms.Compose`类
这个类是一个包装类，可以将上述定义的变换类对象串起来形成一个整体的变换
```python
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

```
---

## 数据装载
上述预处理后的数据集，我们已经可以使用了，但是一般我们会先对数据进行装载，会有以下好处：
    - 批处理
    - 乱序遍历
    - 支持多处理器的并行装载

`torch.utils.data.DataLoader`是一个提供上述功能的类。
```python
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

```
---

> 装载后每批数据会比原本数据集中的每个数据多一个维度。

##　torchvision中国红的通用数据集格式`ImageFolder`
```
root/ants/xxx.png
root/ants/xxy.jpeg
root/ants/xxz.png
.
.
.
root/bees/123.jpg
root/bees/nsdf3.png
root/bees/asd932_.png
```
---
上面"ants"和“bees”等是类标签。transforms中很多操作都是对这种格式的数据进行的。简易的数据集代码如下：
```pythom
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```
---

