#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   net.py    
@Contact :   xxx@163.com
@Desciption :   神经网络
@Author  :zhu
@Modify Time : 2020/3/17 下午11:04       
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
"""
你只需定义 forward 函数, backward 函数(计算梯度)在使用 autograd 时自动为你创建.你可以
在 forward 函数中使用 Tensor 的任何操作。
net.parameters() 返回模型需要学习的参数。
"""
params = list(net.parameters())
print(len(params))
print(params[0].size())

"""
forward 的输入和输出都是 autograd.Variable .注意:这个网络(LeNet)期望的输入大小是
32*32.如果使用MNIST数据集来训练这个网络,请把图片大小重新调整到32*32.
"""
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 将所有参数的梯度缓存清零,然后进行随机梯度的的反向传播.
net.zero_grad()
out.backward(torch.randn(1, 10))

"""
注意
torch.nn 只支持小批量输入,整个 torch.nn 包都只支持小批量样本,而不支持单个样本
例如, nn.Conv2d 将接受一个4维的张量,每一维分别是
(样本数*通道数*高*宽).
如果你有单个样本,只需使用 input.unsqueeze(0) 来添加其它的维数.
"""

"""
回顾
torch.Tensor -支持自动编程操作(如 backward() )的多维数组。同时保持梯度的张
量。
nn.Module -神经网络模块.封装参数,移动到GPU上运行,导出,加载等
nn.Parameter -一种张量,当把它赋值给一个 Module 时,被自动的注册为参数.
autograd.Function -实现一个自动求导操作的前向和反向定义, 每个张量操作都会创建至
少一个 Function 节点,该节点连接到创建张量并对其历史进行编码的函数
"""
"""
损失函数
一个损失函数接受一对(output, target)作为输入(output为网络的输出,target为实际值),计算一个
值来估计网络的输出和目标值相差多少。
在nn包中有几种不同的损失函数.一个简单的损失函数是: nn.MSELoss ,它计算输入和目标之间的
均方误差。

"""
output = net(input)
target = torch.randn(10)
# a dummy target, for example
target = target.view(1, -1)
# make it the same shape as output
criterion = nn.MSELoss()
print(criterion)

loss = criterion(output, target)
print(loss)
"""
现在,你反向跟踪 loss ,使用它的 .grad_fn 属性,你会看到向下面这样的一个计算图:
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view -> linear -> relu ->
linear -> relu -> linear -> MSELoss -> loss
所以, 当你调用 loss.backward() ,整个图被区分为损失以及图中所有具有 requires_grad = Tru
e 的张量,并且其 .grad 张量的梯度累积。
"""

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
"""
反向传播
"""
# loss=criterion(output,target)
# print(loss)

net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 神经网络包包含了各种用来构成深度神经网络构建块的模块和损失函数,一份完整的文档查看[这
# 里]:(https://pytorch.org/docs/nn)
# 更新权重
"""
weight=weight-learning_rate*gradient
"""

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

"""
torch.optim  更新规则
"""
import torch.optim as  optim

# create your optimizer
optimzer = optim.SGD(net.parameters(), lr=0.01)
# in your training loop:
optimzer.zero_grad()  # zero the gradient buffers
output=net(output,target)
loss.backward()
optimzer.step()  # Does the update
