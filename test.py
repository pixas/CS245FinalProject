from models.NGCF import NGCF
from utils.data import AUTHOR_CNT, PAPER_CNT, Data
import torch
import torch.nn as nn

if __name__ == '__main__':
    a = torch.Tensor([[1, 2], [2, 3], [1, 3]])
    print(a >= 2)
    print(torch.sum(a >= 2))
    b = torch.Tensor([[1, 2], [3, 4]])
    # print(a * b)
    d = torch.matmul(a, b.transpose(0, 1))
    print(d)
    # print(d)
    # # print(torch.mul(a, b))
    # # print(torch.sum(torch.mul(a, b), axis=1))
    # print(a)
    c = [[0, 0], [1, 0], [2, 1]]
    # # print(a[c])
    e = d[list(zip(*c))]
    print(e)
    f = [1, 2]
    print(a[f])
    print(torch.norm(a[f]) ** 2)
    print()
    print(nn.LogSigmoid()(e))
    # print(a.shape)
