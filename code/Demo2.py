#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Demo2.py    
@Contact :   xxx@163.com
@Desciption :   
@Author  :zhu
@Modify Time : 2020/3/16 下午11:39       
'''

import numpy as np
import torch

# 张量和数组的转换
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
# 张量的添加
a.add_(1)
print(a)
print(b)

# 数组变张量
a=np.ones(10)

b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
