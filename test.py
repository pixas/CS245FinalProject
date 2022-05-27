from functools import cmp_to_key
import numpy as np
from models.NGCF import NGCF
import torch
import torch.nn as nn
import pickle

if __name__ == '__main__':
    # a = torch.Tensor([[1, -1], [2, 3], [0.1, 3]])
    # b = torch.nn.functional.sigmoid(a)
    # print(b)
    # # print(a >= 2)
    # # print(torch.sum(a >= 2))
    # # b = torch.Tensor([[1, 2], [3, 4]])
    # # # print(a * b)
    # # d = torch.matmul(a, b.transpose(0, 1))
    # # print(d)
    # # # print(d)
    # # # # print(torch.mul(a, b))
    # # # # print(torch.sum(torch.mul(a, b), axis=1))
    # # # print(a)
    # # c = [[0, 0], [1, 0], [2, 1]]
    # # # # print(a[c])
    # # e = d[list(zip(*c))]
    # # print(e)
    # f = [1, 2]
    # print(a[f])
    # print(torch.norm(a[f]) ** 2)
    # print(torch.log(a))
    # # print()
    # # print(nn.LogSigmoid()(e))
    # with open("./data/author_vec.pkl", 'rb') as f:
    #     x = pickle.load(f)
    # print(x.shape)
    arr = np.loadtxt()
    # print(a.shape)
    
