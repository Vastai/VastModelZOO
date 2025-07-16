#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2022/09/16 13:58:22
'''


import scipy.io as scio

dataFile = './testCharBound.mat'
data = scio.loadmat(dataFile)
data_length = len(data["testCharBound"][0])

with open("./test_info.txt", 'w') as fin:
    for index in range(data_length):
        fin.write(data["testCharBound"][0][index][0][0]+" "+data["testCharBound"][0][index][1][0]+ "\n")
