#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Demo3.py    
@Contact :   xxx@163.com
@Desciption :   Autograd 自动求导
@Author  :zhu
@Modify Time : 2020/3/17 下午10:33       
'''

# PyTorch 中所有神经网络的核心是 autograd 包.我
# 张量 (Tensor)

import torch
# torch.Tensor
# torch.Tensor.requires_grad =True,自动跟踪其上所有操作
# torch.Tensor.backward()  自动计算梯度
# torch.Tensor.grad   c存储梯度
# torch.Tensor.detach()将其计算从历史记录中分离出来，防止计算被跟踪

# Tensor和Function互相连接并构建一个非循环图构建一个完整的计算过程。每个张量都有一
# 个 .grad_fn 属性,该属性引用已创建Tensor的Function(除了用户创建的Tensors - 它们的 gr
# ad_fn 为 None )。
# 如果要计算导数,可以在Tensor上调用 .backward() 。如果Tensor是标量(即它包含一个元素
# 数据),则不需要为 backward() 指定任何参数,但是如果它有更多元素,则需要指定一个梯度
# 参数,该参数是匹配Tensor和Function互相连接并构建一个非循环图构建一个完整的计算过程。每个张量都有一
# 个 .grad_fn 属性,该属性引用已创建Tensor的Function(除了用户创建的Tensors - 它们的 gr
# ad_fn 为 None )。
# 如果要计算导数,可以在Tensor上调用 .backward() 。如果Tensor是标量(即它包含一个元素
# 数据),则不需要为 backward() 指定任何参数,但是如果它有更多元素,则需要指定一个梯度
# 参数,该参数是匹配形状的张量。形状的张量。Tensor和Function互相连接并构建一个非循环图构建一个完整的计算过程。每个张量都有一
# 个 .grad_fn 属性,该属性引用已创建Tensor的Function(除了用户创建的Tensors - 它们的 gr
# ad_fn 为 None )。
# 如果要计算导数,可以在Tensor上调用 .backward() 。如果Tensor是标量(即它包含一个元素
# 数据),则不需要为 backward() 指定任何参数,但是如果它有更多元素,则需要指定一个梯度
# 参数,该参数是匹配形状的张量。
# 每个张量都有一个 .grad_fn 属性

x=torch.ones(2,2)
print(x)
y=x+2
print(y)

print(y.grad_fn)
print(x.grad_fn)



x=torch.ones(2,2,requires_grad=True)
print(x)
y=x+2
print(y)
print(y.grad_fn)  #
print(x.grad_fn) #自己创建无
z=y*y*3

out=z.mean()
print(z,out)
# .requires\_grad_(...) 就地更改现有的Tensor的 requires_grad 标志。如果没有给出,输入标志默认为False。

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
print('=====================')
x=torch.ones(2,2,requires_grad=True)

print(x.grad)