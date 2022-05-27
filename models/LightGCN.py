from turtle import forward
import torch
import math

import torch.nn.functional as F

from torch.nn.parameter import Parameter

class LightGCN(torch.nn.Module):
    def __init__(self, dropout, keep_prob, num_layers) -> None:
        super(LightGCN, self).__init__()
        self.dropout_flag = dropout
        self.keep_prob = keep_prob
        self.num_layers = num_layers

    def reset_parameters(self):
        pass
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        x = x.coalesce()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, adj_t, keep_prob):
        graph = self.__dropout_x(adj_t, keep_prob)
        return graph

    def forward(self, x, adj_t):
        if self.dropout_flag:
            # TODO: not drop when testing
            g_dropped = self.__dropout(adj_t, self.keep_prob)
        else:
            g_dropped = adj_t
        
        emb = x
        embs = [x]
        for layer_idx in range(self.num_layers):
            emb = torch.sparse.mm(g_dropped, emb)
            embs.append(emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        print(light_out.shape)
        return light_out
        


        

    