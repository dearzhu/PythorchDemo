#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Type_class.py    
@Contact :   xxx@163.com
@Desciption :    分类器
@Author  :zhu
@Modify Time : 2020/3/21 下午6:59       
'''

"""
数据转换
处理图像,文本,音频和视频数据时,你可以使用标准的Python包来加载数据到一
个numpy数组中.然后把这个数组转换成 torch.*Tensor 。
对于图像,有诸如Pillow,OpenCV包等非常实用
对于音频,有诸如scipy和librosa包
对于文本,可以用原始Python和Cython来加载,或者使用NLTK和SpaCy 对于视觉,我们创建了
一个 torchvision 包,包含常见数据集的数据加载,比如Imagenet,CIFAR10,MNIST等,和图像
转换器,也就是 torchvision.datasets 和 torch.utils.data.DataLoader 。


"""

"""
创建一个图像分类器
步骤进行:
使用 torchvision 加载和归一化CIFAR10训练集和测试集.
定义一个卷积神经网络
定义损失函数
在训练集上训练网络
在测试集上测试网络
"""

import  torch.utils.data.dataloader
import torchvision.transforms as transforms
import torchvision

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloads=torch.utils.data.dataloader(trainset,batch_size=4,sshuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
