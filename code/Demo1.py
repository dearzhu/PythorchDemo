#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Demo1.py    
@Contact :   xxx@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/3/16 下午10:43   zhu      1.0         None
'''
#张量
#张量类似于numpy,但可以通过ＧＰＵ计算

from __future__ import  print_function
import  torch
#构建初始化５＊３矩阵
x=torch.Tensor(5,3)
print(x)

#构建一个零矩阵,使long类型
x=torch.zeros(5,3,dtype=torch.long)
print(x)

#直接构建张量
x=torch.Tensor([5.5,3])
print(x)

x=x.new_ones(5,3,dtype=torch.double)
print(x)

#覆盖类型
x=torch.rand_like(x,dtype=torch.float)
print(x)

#获取张量的大小
print(x.size())  #n元组，支持所有元组操作


#张量的加法
y=torch.rand(5,3)
print(x+y)
print(torch.add(x,y))

# 给出一个输出张量作为参数
result=torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

# 原地操作，
# 吧ｘ加到ｙ上
y.add_(x)
print(y)
# 何在原地(in-place)改变张量的操作都有一个 _ 后缀。例如 x.copy_(y) , x.t_() 操作将改变
print(x[:,-1])

# 调整大小:如果要调整张量/重塑张量,可以使用 torch.view
#
#
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# 的意思是没有指定维度?
# -1
print(x.size(), y.size(), z.size())

# 如果你有一个单元素张量,使用 .item() 将值作为Python数字

x = torch.randn(1)
print(x)
print(x.item())