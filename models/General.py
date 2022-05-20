from typing import List
from matplotlib.style import use
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor 
from argparse import ArgumentParser
from torch_sparse import SparseTensor
from models.NGCF import NGCF
from models.random_walk import RandomWalk
from models.GCN import GNN

class General(nn.Module):
    def __init__(self,Pa_layers:int,Au_layers:int,
                paper_adj:SparseTensor,author_adj:SparseTensor,
                Paperdropout:float,Authordropout:float,
                n_authors: int, n_papers: int, 
                num_layers: int, NGCFembed_dim: int, dropoutNGCF:float,
                paper_dim: int, author_dim: int,
                norm_adj: SparseTensor, layer_size_list: List[int],
                args: ArgumentParser,
                use_pretrain= False) -> None:
        """initializes General model
        Args:
        GCN:
            Pa_input(int) : input dimension of paper
            Au_input(int) : input dimension of author
            Pa_layers(int): paper GNN layers
            Au_layers(int): author GNN layers
            paper_adj(SparseTensor): papaer adjacent matrix
            author_adj(sparseTensor): author adjacent matrix
            Paperdropout(float)
            Authordropout(float)
            use_pretrain: whether to use pretrained author embedding
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
        # self.embed_layer = nn.Embedding(n_authors, author_dim)
        # self.RW = RandomWalk(RWembed_dim, stack_layers, dropoutRW, args)
        self.use_pretrain = use_pretrain
        if not use_pretrain:
            self.auther_emb = nn.Embedding(n_authors,author_dim)
        self.paper_adj = paper_adj
        self.author_adj = author_adj
        
        self.au_GNN = GNN(author_dim,author_dim,author_dim,Au_layers,Authordropout,True)
        self.pa_GNN = GNN(paper_dim,paper_dim,paper_dim,Pa_layers,Paperdropout,True)
        
        
        self.NGCF = NGCF(n_authors, n_papers, dropoutNGCF, 
                 num_layers, NGCFembed_dim, paper_dim, author_dim,
                 norm_adj, layer_size_list)
    
    
    def forward(self, author_embedding: Tensor,
                paper_embedding: Tensor):
        """update General model
        Args:
            author_embedding (Tensor): (N, d), where N is the number of authors
            paper_embedding (Tensor): (M, d), where M is the number of paper
        Returns:
            _type_: _description_
        """
        if not self.use_pretrain:
            author_embedding = self.auther_emb(author_embedding)

        author_embedding = self.au_GNN(author_embedding,self.author_adj)
        paper_embedding = self.pa_GNN(paper_embedding,self.paper_adj)
        author_embedding_new, paper_embedding_new = self.NGCF(author_embedding, paper_embedding)
        
        return author_embedding_new, paper_embedding_new