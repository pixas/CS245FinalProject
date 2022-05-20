import torch
import math
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn.parameter import Parameter


class GraphConv(torch.nn.Module):
    """
    Single graph convolutional layer
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Parameter(torch.FloatTensor(in_channels,out_channels))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = adj@inputs
        output = torch.mm(support,self.W)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GNN, self).__init__()

        # A list of GraphConv layers
        self.convs = [GraphConv(input_dim,hidden_dim)]
        for i in range(num_layers-2):
            self.convs.append(GraphConv(hidden_dim,hidden_dim))
        self.convs.append(GraphConv(hidden_dim,output_dim))

        self.convs = torch.nn.ModuleList(self.convs)
        # A list of 1D batch normalization layers
        self.bns = [torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)]
        self.bns = torch.nn.ModuleList(self.bns)
        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax(1)

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        out = None
        x = self.convs[0](x,adj_t)
        for i in range(len(self.bns)-1):
            x = self.bns[i](x)
            x = F.dropout(x,self.dropout,training=self.training)
            x = F.relu(x)
            x = self.convs[i+1](x,adj_t)
        
        x = self.bns[-1](x)
        x = F.relu(x)
        x = F.dropout(x,self.dropout,training=self.training)
        x = self.convs[-1](x,adj_t)
        if self.return_embeds:
            out = x 
        else:
            out = self.softmax(x)

        return out