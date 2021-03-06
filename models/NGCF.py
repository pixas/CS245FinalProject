from re import A
import torch 
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F
from typing import List 

class NGCF(nn.Module):
    def __init__(self, n_authors: int, n_papers: int, dropout: float, 
                 num_layers: int, embed_dim: int, paper_dim: int, author_dim: int,
                 norm_adj: SparseTensor, layer_size_list: List[int]) -> None:
        """Initializes internal Module state of NGCF.

        Args:
            n_authors (int): number of authors
            n_papers (int): number of paper
            dropout (float): the dropout rate
            num_layers (int): the layer of convolutional layers
            embed_dim (int): the internal embedding dimension for all layers
            paper_dim (int): the dimension for paper embedding. Default to `embed_dim`
            author_dim (int): the dimension for author embedding. Default to `embed_dim`
            norm_adj (SparseTensor): the Laplacian Matrix of adjacent matrix
            n_fold (int): cannot multiply (N+M, N+M) with (N+M, d) directly; split into several (N+M/n_fold, N+M) and (N+M, d) and concatenate them together in the end.
        """
        super(NGCF, self).__init__()
        self.n_authors = n_authors
        self.n_papers = n_papers

        self.num_layers = num_layers
        self.layer_size = [embed_dim] + layer_size_list
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.paper_dim = paper_dim
        self.author_dim = author_dim
        self.norm_adj = norm_adj
        
        assert len(self.layer_size) == self.num_layers + 1
        assert author_dim == paper_dim == embed_dim, "Input embeddings must have the same shape"
         
        self.weights_1 = nn.ModuleList([])
        self.weights_2 = nn.ModuleList([])
        self.relu1 = nn.ModuleList([])
        self.relu2 = nn.ModuleList([])
        for i in range(num_layers):
            self.weights_1.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.weights_2.append(nn.Linear(self.layer_size[i], self.layer_size[i + 1]))
            self.relu1.append(nn.LeakyReLU())
            self.relu2.append(nn.LeakyReLU())
            
            
    def reset_parameters(self):
        for i, module in enumerate(self.weights_1):
            nn.init.xavier_normal_(module.weight, gain=2 ** -0.5)
        
        for i, module in enumerate(self.weights_2):
            nn.init.xavier_normal_(module.weight, gain=2 ** -0.5)
      
    
    def forward(self, author_embedding: Tensor, 
                paper_embedding: Tensor):
        """update author embedding and paper embedding in NGCF layers

        Args:
            author_embedding (Tensor): (N, d): author embedding after undertaking random walk and BiLSTM
            paper_embedding (Tensor): (M, d): paper embedding after undertaking random walk and BiLSTM
        """

        # A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = torch.cat([author_embedding, paper_embedding], 0)
        all_embeddings = [ego_embeddings]
        
        for k in range(self.num_layers):

            side_embeddings = self.norm_adj @ ego_embeddings
            sum_embeddings = self.relu1[k](self.weights_1[k](side_embeddings))
            
            bi_embeddings = ego_embeddings * side_embeddings
            bi_embeddings = self.relu2[k](self.weights_2[k](bi_embeddings))
            
            ego_embeddings = sum_embeddings + bi_embeddings
            
            ego_embeddings = F.dropout(ego_embeddings, self.dropout, self.training)
            
            norm_embeddings = F.normalize(ego_embeddings, 2, 1)
            all_embeddings.append(norm_embeddings)
        
        all_embeddings = torch.cat(all_embeddings, 1)
        author_embedding_new, paper_embedding_new = torch.split(all_embeddings, [self.n_authors, self.n_papers], 0)
        
        return author_embedding_new, paper_embedding_new
            
        
class LightGCN(nn.Module):
    def __init__(self, n_authors: int, n_papers: int, dropout: float, 
                 num_layers: int, norm_adj: SparseTensor) -> None:
        super(LightGCN, self).__init__()
        self.n_authors = n_authors
        self.n_papers = n_papers

        self.num_layers = num_layers
        self.norm_adj = norm_adj
    
    def forward(self, author_embedding: Tensor, 
                paper_embedding: Tensor):
        ego_embedding = torch.cat([author_embedding, paper_embedding], 0)
        embedding_list = [ego_embedding]
        
        for i in range(self.num_layers):
            grouped_embedding = self.norm_adj @ embedding_list[-1]
            embedding_list.append(grouped_embedding)
        embedding_list = [i.unsqueeze(0) for i in embedding_list]
        output_embedding = torch.cat(embedding_list, 0).mean(0)
        output_author_embedding, output_paper_embedding = torch.split(output_embedding, [self.n_authors, self.n_papers], 0)
        return ego_embedding, output_author_embedding, output_paper_embedding

