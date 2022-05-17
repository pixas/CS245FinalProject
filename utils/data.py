import numpy as np
import pickle
import random
import time
import torch
from torch_sparse import SparseTensor
from torch import Tensor 
from typing import Dict, List, Set, Tuple

# Better to be included in a global config file
AUTHOR_GRAPH_PATH = 'data/author_file_ann.txt'
PAPER_GRAPH_PATH = 'data/paper_file_ann.txt'
BIPARTITE_GRAPH_TRAIN_PATH = 'data/bipartite_train_ann.txt'
BIPARTITE_GRAPH_TEST_PATH = 'data/bipartite_test_ann.txt'
PAPER_FEATURE_PATH = 'data/feature.pkl'
AUTHOR_CNT = 6611
PAPER_CNT = 79937

AUTHOR_ADJ_PATH = 'data/author_adj.pkl'
AUTHOR_AUTHOR_MAP_PATH = 'data/author_author_map.pkl'
PAPER_ADJ_PATH = 'data/paper_adj.pkl'
PAPER_PAPER_MAP_PATH = 'data/paper_paper_map.pkl'
BIPARTITE_ADJ_PATH = 'data/bipartite_adj.pkl'
BIPARTITE_LAP_PATH = 'data/bipartite_lap.pkl'
AUTHOR_PAPER_MAP_PATH = 'data/author_paper_map.pkl'

class Data(object):
    def __init__(self, batch_size: int) -> None:
        """Initializes internal Module state of Data.
        Args:
            batch_size: The batch size to use for training.
        """
        self.batch_size = batch_size
        self.n_authors, self.n_papers = AUTHOR_CNT, PAPER_CNT
        self.author_adj_matrix = self.get_author_adj_matrix()
        self.author_author_map = self.get_author_author_map()
        self.paper_adj_matrix = self.get_paper_adj_matrix()
        self.paper_paper_map = self.get_paper_paper_map()
        self.paper_embeddings = self.get_paper_embeddings()
        self.author_paper_map = self.get_author_paper_map()
        self.bipartite_adj_matrix, self.bipartite_lap_matrix = self.get_bipartite_matrix()

    def get_author_author_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from authors to authors in coauthor network.
        Returns:
            The mapping from authors to authors.
        Example:
            author 1 has coauthored with author 10 and author 11
            return {1: {10, 11}, 10: {1}, 11: {1}}
        """
        try:
            t1 = time.time()
            with open(AUTHOR_AUTHOR_MAP_PATH, 'rb') as f:
                author_author_map = pickle.load(f)
            print(f'Load author-author map from {AUTHOR_AUTHOR_MAP_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(AUTHOR_GRAPH_PATH, 'r') as f:
                lines = f.readlines()
                author_author_map = {author: set() for author in range(AUTHOR_CNT)}
                for line in lines:
                    for author, coauthor in [line.strip().split(' ')]:
                        author_author_map[int(author)].add(int(coauthor))
                        author_author_map[int(coauthor)].add(int(author))
            print(f'Build author-author map, time cost: {time.time() - t1: .3f}s')
            with open(AUTHOR_AUTHOR_MAP_PATH, 'wb') as f:
                pickle.dump(author_author_map, f)

        return author_author_map

    def get_author_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the coauthor graph.
        Returns:
            The author adjacency matrix.
        """
        try:
            t1 = time.time()
            with open(AUTHOR_ADJ_PATH, 'rb') as f:
                author_adj_matrix = pickle.load(f)
            print(f'Load author adjacency matrix from {AUTHOR_ADJ_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(AUTHOR_GRAPH_PATH, 'r') as f:
                lines = f.readlines()
                author_author_map = {author: set() for author in range(AUTHOR_CNT)}
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
            with open(AUTHOR_ADJ_PATH, 'wb') as f:
                pickle.dump(author_adj_matrix, f)

        return author_adj_matrix

    def get_paper_paper_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from papers to papers in citation network.
        Returns:
            The mapping from papers to papers.
        Example:
            paper 1 has cited paper 10, 11
            return {1: {10, 11}, 10: {1}, 11: {1}}
        Note: all paper indexes are in range [0, PAPER_CNT)
        """
        try:
            t1 = time.time()
            with open(PAPER_PAPER_MAP_PATH, 'rb') as f:
                paper_paper_map = pickle.load(f)
            print(f'Load paper-paper map from {PAPER_PAPER_MAP_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(PAPER_GRAPH_PATH, 'r') as f:
                lines = f.readlines()
                paper_paper_map = {paper: set() for paper in range(PAPER_CNT)}
                for line in lines:
                    for paper, cited_paper in [line.strip().split(' ')]:
                        paper_paper_map[int(paper)].add(int(cited_paper))
                        paper_paper_map[int(cited_paper)].add(int(paper))
            print(f'Build paper-paper map, time cost: {time.time() - t1: .3f}s')
            with open(PAPER_PAPER_MAP_PATH, 'wb') as f:
                pickle.dump(paper_paper_map, f)

        return paper_paper_map

    def get_paper_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the citation network among papers.
        Returns:
            The paper adjacency matrix.
        """
        try:
            t1 = time.time()
            with open(PAPER_ADJ_PATH, 'rb') as f:
                paper_adj_matrix = pickle.load(f)
            print(f'Load paper adjacency matrix from {PAPER_ADJ_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(PAPER_GRAPH_PATH, 'r') as f:
                lines = f.readlines()
                paper_paper_map = {paper: set() for paper in range(PAPER_CNT)}
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
            with open(PAPER_ADJ_PATH, 'wb') as f:
                pickle.dump(paper_adj_matrix, f)

        return paper_adj_matrix

    def get_paper_embeddings(self) -> torch.Tensor:
        """Returns the initial embedding of paper from feature file generated by USE
        Returns:
            The USE feature of papers.
        """
        t1 = time.time()
        with open(PAPER_FEATURE_PATH, 'rb') as f:
            paper_embeddings = pickle.load(f)
        print(f'Load paper embeddings from {PAPER_FEATURE_PATH}, time cost: {time.time() - t1: .3f}s')
        return paper_embeddings

    def get_bipartite_matrix(self) -> Tuple[SparseTensor, SparseTensor]:
        """Returns the adjacency matrix and Laplacian matrix of the bipartite graph.
        Returns:
            The bipartite adjacency matrix and Laplacian matrix.
        """
        try:
            t1 = time.time()
            with open(BIPARTITE_ADJ_PATH, 'rb') as f:
                bipartite_adj_matrix = pickle.load(f)
            with open(BIPARTITE_LAP_PATH, 'rb') as f:
                bipartite_lap_matrix = pickle.load(f)
            print(f'Load bipartite adjacency matrix and Laplacian matrix from {BIPARTITE_ADJ_PATH} and {BIPARTITE_LAP_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            assert self.author_paper_map != None
            degree_value = [0] * (AUTHOR_CNT + PAPER_CNT)
            index = []
            for author, papers in self.author_paper_map.items():
                degree_value[author] = len(papers)
                for paper in papers:
                    index.append([author, paper])
                    degree_value[paper] += 1
            # adjacency matrix
            v = [1] * len(index)
            bipartite_adj_matrix = torch.sparse_coo_tensor(list(zip(*index)), v, (AUTHOR_CNT + PAPER_CNT, PAPER_CNT + AUTHOR_CNT))

            # Laplacian matrix
            bipartite_lap_index = index
            bipartite_lap_value = [1 / (degree_value[author] * degree_value[paper]) ** (1 / 2) for author, paper in index]
            bipartite_lap_matrix = torch.sparse_coo_tensor(list(zip(*bipartite_lap_index)), bipartite_lap_value, (AUTHOR_CNT + PAPER_CNT, PAPER_CNT + AUTHOR_CNT))
            print(f'Build bipartite adjacency matrix and Laplacian matrix, time cost: {time.time() - t1: .3f}s')
            with open(BIPARTITE_ADJ_PATH, 'wb') as f:
                pickle.dump(bipartite_adj_matrix, f)
            with open(BIPARTITE_LAP_PATH, 'wb') as f:
                pickle.dump(bipartite_lap_matrix, f)
        return bipartite_adj_matrix, bipartite_lap_matrix
    
    def get_author_paper_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from authors to papers.
        Returns:
            The mapping from authors to papers.
        Example:
            author 1 has cited paper 10, 11
            author 2 has cited paper 10, 12
            return {1: [10, 11], 2: [10, 12]}
        Note: all paper indexes have been added AUTHOR_CNT
        """
        try:
            t1 = time.time()
            with open(AUTHOR_PAPER_MAP_PATH, 'rb') as f:
                author_paper_map = pickle.load(f)
            print(f'Load author-paper map from {AUTHOR_PAPER_MAP_PATH}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(BIPARTITE_GRAPH_TRAIN_PATH, 'r') as f:
                lines = f.readlines()
                author_paper_map = {author: set() for author in range(AUTHOR_CNT)}
                for line in lines:
                    for author, paper in [line.strip().split(' ')]:
                        author_paper_map[int(author)].add(int(paper) + AUTHOR_CNT)
            print(f'Build author-paper map, time cost: {time.time() - t1: .3f}s')
            with open(AUTHOR_PAPER_MAP_PATH, 'wb') as f:
                pickle.dump(author_paper_map, f)
        return author_paper_map

    def sample(self) -> Tuple[List[int], List[int]]:
        """
        Sample several pairs of authors and papers with batch size
        Returns:
            The sampled authors and papers.
        Example:
            author 1 has cited paper 10, 11, not cited paper 12
            author 2 has cited paper 10, 12, not cited paper 11
            return 
                authors:    [1, 2]
                pos_papers: [10, 12] 
                neg_papers: [12, 11]
        """
        authors = random.sample(range(self.n_authors), self.batch_size)

        def sample_pos_papers_for_author(author: int, num: int) -> List[int]:
            pos_papers = list(self.author_paper_map[author])
            pos_papers_num = len(pos_papers)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                random_idx = np.random.randint(low=0, high=pos_papers_num, size=1)[0]
                pos_paper_idx = pos_papers[random_idx]
                if pos_paper_idx not in pos_batch:
                    pos_batch.append(pos_paper_idx)
            return pos_batch
        
        def sample_neg_papers_for_author(author: int, num: int) -> List[int]:
            neg_papers = list(set(range(AUTHOR_CNT, AUTHOR_CNT + self.n_papers)) - set(self.author_paper_map[author]))
            neg_papers_num = len(neg_papers)
            neg_batch = []
            while True:
                if len(neg_batch) == num:
                    break
                random_idx = np.random.randint(low=0, high=neg_papers_num, size=1)[0]
                neg_paper_idx = neg_papers[random_idx]
                if neg_paper_idx not in neg_batch:
                    neg_batch.append(neg_paper_idx)
            return neg_batch
        
        pos_papers, neg_papers = [], []
        for author in authors:
            pos_papers.extend(sample_pos_papers_for_author(author, 1))
            neg_papers.extend(sample_neg_papers_for_author(author, 1))
        
        return authors, pos_papers, neg_papers
    
    def generate_random_walk_author(self, t: int) -> Tensor:
        """Generate the random walk for all authors in coauthor network.

        Args:
            t (int): the length of random walk (t << N).
        Returns:
            random_walk_matrix (Tensor): (AUTHOR_CNT, t)
        """
        random_walk_matrix = torch.zeros(size=(AUTHOR_CNT, t), dtype=torch.int32)
        for author in range(AUTHOR_CNT):
            random_walk_matrix[author, 0] = author
            pre = author
            for i in range(1, t):
                cur = random.choice(list(self.author_author_map[pre]))
                random_walk_matrix[author, i] = cur
                pre = cur
        return random_walk_matrix

    def generate_random_walk_paper(self, t: int) -> Tensor:
        """Generate the random walk for all papers in citation network.

        Args:
            t (int): the length of random walk (t << M).
        Returns:
            random_walk_matrix (Tensor): (PAPER_CNT, t)
        """
        random_walk_matrix = torch.zeros(size=(PAPER_CNT, t), dtype=torch.int32)
        for paper in range(PAPER_CNT):
            random_walk_matrix[paper, 0] = paper
            pre = paper
            for i in range(1, t):
                cur = random.choice(list(self.paper_paper_map[pre]))
                random_walk_matrix[paper, i] = cur
                pre = cur
        return random_walk_matrix

if __name__ == '__main__':
    data_generator = Data(batch_size=10)
    # print(data_generator.author_author_map[0])
    # print(data_generator.paper_paper_map[0])
    print('--author--')
    path = data_generator.generate_random_walk_author(t=5)[0]
    for i in path:
        print(i, data_generator.author_author_map[int(i)])
    print('--paper--')
    path = data_generator.generate_random_walk_paper(t=5)[0]
    for i in path:
        print(i, data_generator.paper_paper_map[int(i)])
    # print(data_generator.author_adj_matrix[0])
    # print(data_generator.paper_adj_matrix[0])
    # print(data_generator.bipartite_adj_matrix[0])