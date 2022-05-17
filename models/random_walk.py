import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor 
from argparse import ArgumentParser
class RandomWalk(nn.Module):
    def __init__(self, embed_dim: int, stack_layers: int, dropout: float,
                 args: ArgumentParser) -> None:
        """initializes the update process of random walk process

        Args:
            embed_dim (int): embedding size for all hidden states and embeddings
            stack_layers (int): the number of LSTM or attention modules 
            dropout (float): dropout rate for internal modules
            args (ArgumentParser): the argparser instance containing some necessary arguments
        """
        super(RandomWalk, self).__init__()
        self.embed_dim = embed_dim
        self.stack_layers = stack_layers
        self.dropout = dropout
        
        self.out_paper_norm = nn.LayerNorm(self.embed_dim)
        self.out_author_norm = nn.LayerNorm(self.embed_dim)
        self.args = args
        self.module_type = args.module_type
        assert self.module_type == 'lstm' or self.module_type == 'LSTM' or self.module_type == 'attention', "Only support BiLSTM or Self-Attention"
        
        if self.module_type == 'attention':
            self.num_heads = args.num_heads 
            self.layers_paper = nn.ModuleList([
                nn.MultiheadAttention(embed_dim, self.num_heads,
                                      self.dropout, batch_first=True)
            for i in range(self.stack_layers)])
            self.layers_author = nn.ModuleList([
                nn.MultiheadAttention(embed_dim, self.num_heads,
                                      self.dropout, batch_first=True)
            for i in range(self.stack_layers)])
        else:
            self.layer_author = nn.LSTM(input_size=embed_dim,
                                 hidden_size=embed_dim,
                                 num_layers=self.stack_layers,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=self.dropout)
            self.layer_paper = nn.LSTM(input_size=embed_dim,
                                 hidden_size=embed_dim,
                                 num_layers=self.stack_layers,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=self.dropout)
    
    def _tensor_gather(self, x: Tensor, idx: Tensor):
        idx = idx.repeat((1, 1, self.embed_dim))
        selected_tensor = x.gather(1, idx)
        return selected_tensor
    
    def forward(self, author_embedding: Tensor,
                paper_embedding: Tensor,
                author_selected_idx: Tensor, 
                paper_selected_idx: Tensor):
        """update author embedding and paper_embedding with LSTM or Attention

        Args:
            author_embedding (Tensor): (batch_size, N, d), where N is the number of authors
            paper_embedding (Tensor): (batch_size, M, d), where M is the number of paper
            author_selected_idx (Tensor): (batch_size, T), where T is the length of random walk (T << N)
            paper_selected_idx (Tensor): (batch_size, T), where T is the length of random walk (T << M)

        Returns:
            _type_: _description_
        """
        paper_selected_idx = paper_selected_idx.repeat((1, 1, self.embed_dim))
        author_selected_idx = author_selected_idx.repeat((1, 1, self.embed_dim))
        
        paper_selected_embedding = paper_embedding.gather(1, paper_selected_idx)
        author_selected_embedding = author_embedding.gather(1, author_selected_idx)
        
        if self.module_type == 'attention':
            for i, layer in enumerate(self.layers_paper):
                paper_selected_embedding, _ = layer(paper_selected_embedding,
                                                    paper_selected_embedding,
                                                    paper_selected_embedding)
            for i, layer in enumerate(self.layers_author):
                author_selected_embedding, _ = layer(author_selected_embedding,
                                                     author_selected_embedding,
                                                     author_selected_embedding)
        else:
            paper_selected_embedding = self.layer_paper(paper_selected_embedding)
            author_selected_embedding = self.layer_author(author_selected_embedding)
        
        paper_selected_embedding = self.out_paper_norm(paper_selected_embedding)
        author_selected_embedding = self.out_author_norm(author_selected_embedding)
        
        author_embedding.scatter_(1, author_selected_idx, author_selected_embedding)
        paper_embedding.scatter_(1, paper_selected_idx, paper_selected_embedding)
        
        return author_embedding, paper_embedding
        


if __name__ == "__main__":
    x = torch.randn((2, 10, 8))
    y = torch.randint(0, 7, (2, 5, 1))
    y = y.repeat((1, 1, 8))
    src = torch.randn((2, 5, 8))
    print(x, y, src)
    x.scatter_(1, y, src)
    print(x, y, src)