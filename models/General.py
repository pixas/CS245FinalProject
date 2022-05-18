import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor 
from argparse import ArgumentParser
from torch_sparse import SparseTensor
from models.NGCF import NGCF
from models.random_walk import RandomWalk

class General(nn.Module):
    def __init__(self, RWembed_dim: int, stack_layers: int, dropoutRW: float,
                n_authors: int, n_papers: int, 
                num_layers: int, NGCFembed_dim: int, dropoutNGCF:float,
                paper_dim: int, author_dim: int,
                norm_adj: SparseTensor, n_fold: int,
                args: ArgumentParser) -> None:
        """initializes General model
        Args:
        RW:
            RWembed_dim (int): embedding size for all hidden states and embeddings
            stack_layers (int): the number of LSTM or attention modules 
            dropoutRW (float): dropout rate for internal modules
            args (ArgumentParser): the argparser instance containing some necessary arguments
        NGCF:
            n_authors (int): number of authors
            n_papers (int): number of paper
            dropoutNGCF (float): the dropout rate
            num_layers (int): the layer of convolutional layers
            NGCFembed_dim (int): the internal embedding dimension for all layers
            paper_dim (int): the dimension for paper embedding. Default to `embed_dim`
            author_dim (int): the dimension for author embedding. Default to `embed_dim`
            norm_adj (SparseTensor): the Laplacian Matrix of adjacent matrix
            n_fold (int): cannot multiply (N+M, N+M) with (N+M, d) directly; split into several (N+M/n_fold, N+M) and (N+M, d) and concatenate them together in the end.
        """
        super(General, self).__init__()
        self.RW = RandomWalk(RWembed_dim, stack_layers, dropoutRW, args)
        self.NGCF = NGCF(n_authors, n_papers, dropoutNGCF, 
                 num_layers, NGCFembed_dim, paper_dim, author_dim,
                 norm_adj, n_fold)
    
    
    def forward(self, author_embedding: Tensor,
                paper_embedding: Tensor,
                author_selected_idx: Tensor, 
                paper_selected_idx: Tensor):
        """update General model
        Args:
            author_embedding (Tensor): (N, d), where N is the number of authors
            paper_embedding (Tensor): (M, d), where M is the number of paper
            author_selected_idx (Tensor): (T, 1), where T is the length of random walk (T << N)
            paper_selected_idx (Tensor): (T, 1), where T is the length of random walk (T << M)
        Returns:
            _type_: _description_
        """
        author_embedding, paper_embedding = self.RW(author_embedding, paper_embedding, author_selected_idx, paper_selected_idx)
        author_embedding_new, paper_embedding_new = self.NGCF(author_embedding, paper_embedding)
        
        return author_embedding_new, paper_embedding_new