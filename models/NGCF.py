import torch 
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, n_authors: int, n_papers: int, dropout: float, 
                 num_layers: int, embed_dim: int, paper_dim: int, author_dim: int,
                 norm_adj: SparseTensor, n_fold: int) -> None:
        super(NGCF, self).__init__()
        self.n_authors = n_authors
        self.n_papers = n_papers
        self.n_fold = n_fold 
        self.num_layers = num_layers
        
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.paper_dim = paper_dim
        self.author_dim = author_dim
        self.norm_adj = norm_adj
        
        
        assert author_dim == paper_dim == embed_dim, "Input embeddings must have the same shape"
         
        self.weights_1 = nn.ModuleList([])
        self.weights_2 = nn.ModuleList([])
        self.relu1 = nn.ModuleList([])
        self.relu2 = nn.ModuleList([])
        for i in range(num_layers):
            self.weights_1.append(nn.Linear(embed_dim, embed_dim))
            self.weights_2.append(nn.Linear(embed_dim, embed_dim))
            self.relu1.append(nn.LeakyReLU())
            self.relu2.append(nn.LeakyReLU())
            
            
    def reset_parameters(self):
        for i, module in enumerate(self.weights_1):
            nn.init.xavier_normal_(module.weight, gain=2 ** -0.5)
        
        for i, module in enumerate(self.weights_2):
            nn.init.xavier_normal_(module.weight, gain=2 ** -0.5)
    
    def _split_A_hat(self, X: SparseTensor):
        A_fold_hat = []

        fold_len = (self.n_authors + self.n_papers) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_authors + self.n_papers
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(X[start:end])
        return A_fold_hat    
    
    def forward(self, author_embedding: Tensor, 
                paper_embedding: Tensor):
        """update author embedding and paper embedding in NGCF layers

        Args:
            author_embedding (Tensor): (N, d): author embedding after undertaking random walk and BiLSTM
            paper_embedding (Tensor): (M, d): paper embedding after undertaking random walk and BiLSTM
        """
        A_fold_hat = self._split_A_hat(self.norm_adj)
        ego_embeddings = torch.cat([author_embedding, paper_embedding], 0)
        all_embeddings = [ego_embeddings]
        
        for k in range(self.num_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(A_fold_hat[f] @ ego_embeddings)
            
            side_embeddings = torch.cat(temp_embed, 0)
            sum_embeddings = self.relu1[k](self.weights_1[k](side_embeddings))
            
            bi_embeddings = ego_embeddings * side_embeddings
            bi_embeddings = self.relu2[k](self.weights_2[k](bi_embeddings))
            
            ego_embeddings = sum_embeddings + bi_embeddings
            
            ego_embeddings = F.dropout(ego_embeddings, self.dropout, self.training)
            
            norm_embeddings = F.normalize(ego_embeddings, 2, 1)
            all_embeddings.append(norm_embeddings)
        
        all_embeddings = torch.cat(all_embeddings, 1)
        author_embedding_new, paper_embedding_new = torch.split(all_embeddings, [self.n_authors, self.n_papers, 0])
        
        return author_embedding_new, paper_embedding_new
            
        
        