import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor

class GATBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super(GATBlock, self).__init__()
        self.embed_dim  = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout 
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear1.weight, 2 ** -0.5)
        nn.init.xavier_normal_(self.linear2.weight, 2 ** -0.5)
        
    
    def forward(self, x: Tensor):
        key_padding_mask = torch.all(x == 0, -1)
        residual = x 
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask
        )
        x = self.layer_norm(x + residual)
        residual = x
        x = self.linear2(F.gelu(self.linear1(x)))
        x = self.out_norm(x + residual)
        return x

class GAT(nn.Module):
    def __init__(self, embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 stack_layers: int) -> None:
        super(GAT, self).__init__()
        
        self.layers = nn.ModuleList([
            GATBlock(embed_dim, num_heads, dropout)
        for i in range(stack_layers)])
    
    def forward(self, x: Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        return x[:, 0, :]


class GraphSage(nn.Module):
    def __init__(self, embed_dim: int,
                 stack_layers: int,
                 dropout: int,
                 aggr_way: str = 'LSTM') -> None:
        super(GraphSage, self).__init__()
        self.stack_layers = stack_layers
        self.dropout = dropout
        if aggr_way == 'LSTM':
            self.lstm = nn.LSTM(input_size=embed_dim,
                                 hidden_size=embed_dim,
                                 num_layers=self.stack_layers,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=self.dropout)
            
            pass
        else:
            raise NotImplementedError

        self.act = nn.ReLU()
    
    def forward(self, x: Tensor):
        # x: [B, 1 + #neighbor, d]
        x, (_, _) = self.lstm(x)
        x = x.reshape(x.shape[0], x.shape[1], self.stack_layers, -1)
        x, _ = torch.max(x, dim=-2)
        return x[:, 0, :]