from models.NGCF import NGCF
from utils.data import AUTHOR_CNT, PAPER_CNT, Data
import torch
import torch.nn as nn
import pickle

if __name__ == '__main__':
    score_matrix = torch.randn((4,5, 2))
    idx = [[0, 0],
           [1, 1],
           [2, 2]]
    print(list(zip(*idx)))
    # print(score_matrix[idx[0], idx[1], :].shape)
    # print(a.shape)
