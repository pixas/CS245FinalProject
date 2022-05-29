from typing import List

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor 
from argparse import ArgumentParser
from torch_sparse import SparseTensor
from models.GraphAggregate import GAT, GraphSage
from models.NGCF import NGCF
from models.random_walk import RandomWalk
from models.GCN import GNN
import numpy as np

class General(nn.Module):
    def __init__(self,Pa_layers:int,Au_layers:int,
                paper_adj:SparseTensor,author_adj:SparseTensor,
                Paperdropout:float,Authordropout:float,
                n_authors: int, n_papers: int, 
                only_feature: bool,
                num_layers: int, NGCFembed_dim: int, dropoutNGCF:float,
                paper_dim: int, author_dim: int, layer_size_list: List[int],
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
        self.only_feature = only_feature
        if not use_pretrain:
            self.auther_emb = nn.Embedding(n_authors,author_dim)
        if not only_feature:
            self.paper_emb = nn.Embedding(n_papers, paper_dim)
            self.compress = nn.Linear(paper_dim * 2, paper_dim)
        self.paper_adj = paper_adj
        self.author_adj = author_adj
        
        # self.au_GNN = GNN(author_dim,author_dim,author_dim,Au_layers,Authordropout,True)
        # self.pa_GNN = GNN(paper_dim,paper_dim,paper_dim,Pa_layers,Paperdropout,True)
        # self.pa_sage = GraphSage(embed_dim=paper_dim, stack_layers=Pa_layers, dropout=Paperdropout)
        self.au_GAT = GAT(author_dim, args.num_heads, Authordropout, args.gat_layers)
        self.pa_GAT = GAT(paper_dim, args.num_heads, Paperdropout, args.gat_layers)

    
    def forward(self, author_embedding: Tensor,
                paper_embedding: Tensor,
                paper_feature: Tensor,
                batch_paper_index: List[int],
                batch_author_index: List[int]=None,
                paper_paper_map: List[List[int]]=None,
                paper_padding_mask: Tensor=None,
                author_author_map: List[List[int]]=None,
                author_padding_mask: Tensor=None):
        """update General model
        Args:
            author_embedding (Tensor): (N, d), where N is the number of authors
            paper_embedding (Tensor): (M, d), where M is the number of paper
        Returns:
            _type_: _description_
        """
        if not self.use_pretrain:
            author_embedding = self.auther_emb(author_embedding)
        if not self.only_feature:
            paper_emb = self.paper_emb(paper_embedding)
            fused_paper_emb = torch.cat([paper_emb, paper_feature], -1)
            paper_embedding = self.compress(fused_paper_emb)
        else:
            paper_embedding = paper_feature
            
        # author_embedding_new = self.au_GNN(author_embedding, self.author_adj)
        # paper_embedding_new = paper_embedding
        author_embedding_new = self.batch_gat_layer(
            author_author_map,
            author_padding_mask,
            batch_author_index,
            author_embedding,
            self.au_GAT
        )
        paper_embedding_new = self.batch_gat_layer(paper_paper_map,
                                                   paper_padding_mask,
                                                   batch_paper_index,
                                                   paper_embedding,
                                                   self.pa_GAT)
        # interact_prob = torch.einsum("nd,md->nm", author_embedding_new, paper_embedding_new)
        # interact_prob = torch.sigmoid(interact_prob)
        return author_embedding_new, paper_embedding_new

    def batch_gat_layer(self, 
                        inter_map: List[List[int]],
                        padding_mask: Tensor,
                        batch_index: List[int],
                        embedding: Tensor,
                        layer: GAT):
        batch_interact = inter_map[batch_index]
        batch_mask = padding_mask[batch_index]
        B, K = batch_interact.shape
        batch_interact = np.reshape(batch_interact, (-1, 1))
        batch_embedding = embedding[batch_interact].reshape(B, K, -1)
        batch_embedding = batch_embedding * batch_mask.unsqueeze(-1)
        
        batch_query = embedding[batch_index].unsqueeze(1)
        batch_gat_embedding = layer(batch_query, batch_embedding, batch_mask)
        # batch_embedding_new = embedding.scatter(0,
        #                                         torch.tensor(batch_index,
        #                                                      dtype=torch.int64,
        #                                                      device=embedding.device).unsqueeze(-1).repeat(
        #                                                          1,
        #                                                          embedding.shape[-1]
        #                                                      ), batch_gat_embedding
        # )
        return batch_gat_embedding