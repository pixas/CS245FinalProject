import sys
import numpy as np
import pickle
import random
import time
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import Tensor 
from typing import Dict, List, Set, Tuple


class PrepareData(object):
    def __init__(self, path = 'path') -> None:
        """Initializes internal Module state of Data.
        Args:
            batch_size: The batch size of the data.
            random_walk_length: The length of random walk to use for training.
            device: The device to use for training.
            train_ratio: The ratio of training data.
            path: The path to the data.
        """



        # data path begin #
        self.author_cnt = 6611
        self.paper_cnt = 79937

        self.author_graph_path = f'{path}/author_file_ann.txt'
        self.paper_graph_path = f'{path}/paper_file_ann.txt'
        self.bipartite_graph_train_path = f'{path}/bipartite_train_ann.txt'
        self.bipartite_graph_test_path = f'{path}/bipartite_test_ann.txt'
        self.author_feature_path = f'{path}/author_vec.pkl'
        self.paper_feature_path = f'{path}/feature.pkl'

        self.author_adj_path = f'{path}/author_adj.pkl'
        self.author_author_map_path = f'{path}/author_author_map.pkl'
        self.paper_adj_path = f'{path}/paper_adj.pkl'
        self.paper_paper_map_path = f'{path}/paper_paper_map.pkl'
        self.paper_paper_nei_path = f'{path}/paper_paper_nei.pkl'
        self.paper_mask = f'{path}/paper_mask.npy'
        self.bipartite_adj_path = f'{path}/bipartite_adj.pkl'
        self.bipartite_lap_path = f'{path}/bipartite_lap.pkl'

        self.author_paper_map_path = f'{path}/author_paper_map.pkl'
        self.train_idx_path = f'{path}/train_idx.pkl'
        self.train_authors_path = f'{path}/train_authors.pkl'
        self.train_papers_path = f'{path}/train_papers.pkl'
        
        # data path end #

        self.n_authors, self.n_papers = self.author_cnt, self.paper_cnt
        # self.train_index, self.train_authors, self.train_papers = self.get_train_idx()
        # random.shuffle(self.train_index)
        # self.real_train_index = self.train_index[:int(len(self.train_index) * train_ratio)]
        # self.real_test_index = self.train_index[int(len(self.train_index) * train_ratio):]
        # self.total_train_cnt = len(self.train_index)





    def get_author_author_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from authors to authors in coauthor network.
        Returns:
            The mapping from authors to authors.
        Example:
            author 1 has coauthored with author 10 and author 11
            return {1: {10, 11}, 10: {1}, 11: {1}}
        """

        t1 = time.time()
        with open(self.author_graph_path, 'r') as f:
            lines = f.readlines()
            author_author_map = [list() for author in range(self.author_cnt)]
            for line in lines:
                for author, coauthor in [line.strip().split(' ')]:
                    author_author_map[int(author)].append(int(coauthor))
                    author_author_map[int(coauthor)].append(int(author))
            
            maxl = max([len(coauthors) for coauthors in author_author_map])
            author_padding_mask = np.zeros((self.author_cnt, maxl))
            for i in range(len(author_author_map)):
                author_author_map[i] = author_author_map[i] + [0] * (maxl - len(author_author_map[i]))
                author_padding_mask[i, : len()]
        print(f'Build author-author map, time cost: {time.time() - t1: .3f}s')
        with open(self.author_author_map_path, 'wb') as f:
            pickle.dump(author_author_map, f)
        return author_author_map

    def get_author_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the coauthor graph.
        Returns:
            The author adjacency matrix.
        """

        t1 = time.time()
        with open(self.author_graph_path, 'r') as f:
            lines = f.readlines()
            author_author_map = {author: set() for author in range(self.author_cnt)}
            for line in lines:
                for author, coauthor in [line.strip().split(' ')]:
                    author_author_map[int(author)].add(int(coauthor))
                    author_author_map[int(coauthor)].add(int(author))
            index = []
            for author, coauthors in author_author_map.items():
                for coauthor in coauthors:
                    index.append((author, coauthor))
            v = [1] * len(index)
            author_adj_matrix = torch.sparse_coo_tensor(list(zip(*index)), v, (self.n_authors, self.n_authors))
        print(f'Build author adjacency matrix, time cost: {time.time() - t1: .3f}s')
        with open(self.author_adj_path, 'wb') as f:
            pickle.dump(author_adj_matrix, f)

        return author_adj_matrix

    # def get_paper_paper_nei(self):

    #     t1 = time.time()
        
    #     paper_feature = self.paper_embeddings
    #     paper_paper_map = self.paper_paper_map

    #     # max_connection = max(len(list(x)) for i, x in paper_paper_map.items())
    #     sample_number = 64

    #     neighbor_embedding = torch.zeros((self.paper_cnt, sample_number + 1, paper_feature.shape[-1]), dtype=paper_feature.dtype)
    #     for i, x in paper_paper_map.items():
    #         possible_idx = list(x)
    #         random.shuffle(possible_idx)
    #         idx = torch.tensor(possible_idx if len(possible_idx) <= sample_number else possible_idx[:sample_number], dtype=torch.int64)

    #         gather_idx = idx.unsqueeze(-1).repeat((1, paper_feature.shape[-1]))

    #         gathered_embedding = paper_feature.gather(0, gather_idx)
    #         neighbor_embedding[i] = neighbor_embedding[i].scatter(0, torch.arange(1, len(idx) + 1, 1, dtype=torch.int64).unsqueeze(-1).repeat((1, paper_feature.shape[-1])), gathered_embedding)
    #         neighbor_embedding[i, 0] = paper_feature[i]

    #     print(f'Build paper paper neighborhood, time cost: {time.time() - t1:.3f}')
    #     torch.save(neighbor_embedding, self.paper_paper_nei_path)
    #         # with open(self.paper_paper_nei_path, 'wb') as f:
    #         #     pickle.dump(neighbor_embedding, f)
        

                
    
    def get_paper_paper_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from papers to papers in citation network.
        Returns:
            The mapping from papers to papers.
        Example:
            paper 1 has cited paper 10, 11
            return {1: {10, 11}, 10: {1}, 11: {1}}
        Note: all paper indexes are in range [0, self.paper_cnt)
        """

        t1 = time.time()
        with open(self.paper_graph_path, 'r') as f:
            lines = f.readlines()
            paper_paper_map = [list() for paper in range(self.paper_cnt)]
            for line in lines:
                for paper, cited_paper in [line.strip().split(' ')]:
                    paper_paper_map[int(paper)].append(int(cited_paper))
                    paper_paper_map[int(cited_paper)].append(int(paper))
            
            
            maxl = max([len(cited_papers) for cited_papers in paper_paper_map])
            padding_mask = np.zeros((self.paper_cnt, maxl))
            for i in range(len(paper_paper_map)):
                padding_mask[i, :len(paper_paper_map[i])] = 1
                paper_paper_map[i] = paper_paper_map[i] + [0]*(maxl-len(paper_paper_map[i]))
            
            paper_paper_map = np.array(paper_paper_map)
                    
        print(f'Build paper-paper map, time cost: {time.time() - t1: .3f}s')
        np.save(self.paper_mask, padding_mask)
        with open(self.paper_paper_map_path, 'wb') as f:
            pickle.dump(paper_paper_map, f)


    def get_paper_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the citation network among papers.
        Returns:
            The paper adjacency matrix.
        """

        t1 = time.time()
        with open(self.paper_graph_path, 'r') as f:
            lines = f.readlines()
            paper_paper_map = {paper: set() for paper in range(self.paper_cnt)}
            for line in lines:
                for paper, cited_paper in [line.strip().split(' ')]:
                    paper_paper_map[int(paper)].add(int(cited_paper))
                    paper_paper_map[int(cited_paper)].add(int(paper))
            index = []
            for paper, cited_papers in paper_paper_map.items():
                for cited_paper in cited_papers:
                    index.append((paper, cited_paper))
            v = [1] * len(index)
            paper_adj_matrix = torch.sparse_coo_tensor(list(zip(*index)), v, (self.n_papers, self.n_papers))
        print(f'Build paper adjacency matrix, time cost: {time.time() - t1: .3f}s')
        with open(self.paper_adj_path, 'wb') as f:
            pickle.dump(paper_adj_matrix, f)



    def get_train_idx(self) -> Tuple[List[List[int]], List[int], List[int]]:
        """Returns the training pairs, author indexes and paper indexes.
        Returns:
            train_idx(List[List[int]]): (train_cnt, 2).
            train_author_idx(List[int]): All authors id in training set.
            train_paper_idx(List[int]): All papers id in training set.
        Note:
            all paper indexes are in range [0, self.paper_cnt)
        Example:
            train_idx = [[1, 2], [3, 4], [5, 6]]
            train_author_idx = [1, 3, 5]
            train_paper_idx = [2, 4, 6]
        """

        t1 = time.time()
        with open(self.bipartite_graph_train_path, 'r') as f:
            lines = f.readlines()
            train_authors = set()
            train_papers = set()
            train_idx = []
            for line in lines:
                for author, paper in [line.strip().split(' ')]:
                    train_authors.add(int(author))
                    train_papers.add(int(paper))
                    train_idx.append([int(author), int(paper)])
            train_authors = list(train_authors)
            train_papers = list(train_papers)
        print(f'Build training data, time cost: {time.time() - t1: .3f}s')
        with open(self.train_idx_path, 'wb') as f:
            pickle.dump(train_idx, f)
        with open(self.train_authors_path, 'wb') as f:
            pickle.dump(train_authors, f)
        with open(self.train_papers_path, 'wb') as f:
            pickle.dump(train_papers, f)


    def get_bipartite_matrix(self) -> Tuple[SparseTensor, SparseTensor]:
        """Returns the adjacency matrix and Laplacian matrix of the bipartite graph.
        Returns:
            The bipartite adjacency matrix and Laplacian matrix.
        """
        bipartite_adj_matrix = []
        bipartite_lap_index = []

        t1 = time.time()
        assert self.author_paper_map != None
        degree_value = [0] * (self.author_cnt + self.paper_cnt)
        index = []
        for author, papers in self.author_paper_map.items():
            degree_value[author] = len(papers)
            for paper in papers:
                index.append([author, paper])
                degree_value[paper] += 1
        # adjacency matrix
        v = [1] * len(index)
        bipartite_adj_matrix = torch.sparse_coo_tensor(list(zip(*index)), v, (self.author_cnt + self.paper_cnt, self.paper_cnt + self.author_cnt))

        # Laplacian matrix
        bipartite_lap_index = index
        bipartite_lap_value = [1 / (degree_value[author] * degree_value[paper]) ** (1 / 2) for author, paper in index]
        bipartite_lap_matrix = torch.sparse_coo_tensor(list(zip(*bipartite_lap_index)), bipartite_lap_value, (self.author_cnt + self.paper_cnt, self.paper_cnt + self.author_cnt))

        print(f'Build bipartite adjacency matrix and Laplacian matrix, time cost: {time.time() - t1: .3f}s')
        with open(self.bipartite_adj_path, 'wb') as f:
            pickle.dump(bipartite_adj_matrix, f)
        with open(self.bipartite_lap_path, 'wb') as f:
            pickle.dump(bipartite_lap_matrix, f)

    
    def get_author_paper_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from authors to papers.
        Returns:
            The mapping from authors to papers.
        Example:
            author 1 has cited paper 10, 11
            author 2 has cited paper 10, 12
            return {1: [10, 11], 2: [10, 12]}
        Note: all paper indexes have been added self.author_cnt
        """

        t1 = time.time()
        with open(self.bipartite_graph_train_path, 'r') as f:
            lines = f.readlines()
            author_paper_map = {author: set() for author in range(self.author_cnt)}
            for line in lines:
                for author, paper in [line.strip().split(' ')]:
                    author_paper_map[int(author)].add(int(paper) + self.author_cnt)
        print(f'Build author-paper map, time cost: {time.time() - t1: .3f}s')
        with open(self.author_paper_map_path, 'wb') as f:
            pickle.dump(author_paper_map, f)

        self.author_paper_map = author_paper_map


    def prepare_all(self):
        self.get_author_adj_matrix()
        self.get_author_author_map()
        self.get_paper_adj_matrix()
        self.get_paper_paper_map()
        # self.paper_embeddings = self.get_paper_embeddings()
        self.get_author_paper_map()
        self.get_bipartite_matrix()
        self.get_train_idx()
        
        with open(self.author_feature_path, 'rb') as f:
            self.author_embeddings = pickle.load(f)

        with open(self.paper_feature_path, 'rb') as f:
            self.paper_embeddings = pickle.load(f)


        # self.get_paper_paper_nei()

if __name__ == "__main__":
    output_dir = sys.argv[1]
    x = PrepareData(path=output_dir)
    x.prepare_all()