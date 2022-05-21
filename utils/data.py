import numpy as np
import pickle
import random
import time
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import Tensor 
from typing import Dict, List, Set, Tuple

class Data(object):
    def __init__(self, batch_size: int, random_walk_length: int, device: str, train_ratio: float = 0.9, path = 'path') -> None:
        """Initializes internal Module state of Data.
        Args:
            batch_size: The batch size of the data.
            random_walk_length: The length of random walk to use for training.
            device: The device to use for training.
            train_ratio: The ratio of training data.
            path: The path to the data.
        """
        self.batch_size = batch_size
        self.random_walk_length = random_walk_length
        self.device = device

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
        self.bipartite_adj_path = f'{path}/bipartite_adj.pkl'
        self.bipartite_lap_path = f'{path}/bipartite_lap.pkl'

        self.author_paper_map_path = f'{path}/author_paper_map.pkl'
        self.train_idx_path = f'{path}/train_idx.pkl'
        self.train_authors_path = f'{path}/train_authors.pkl'
        self.train_papers_path = f'{path}/train_papers.pkl'
        
        # data path end #

        self.n_authors, self.n_papers = self.author_cnt, self.paper_cnt
        self.train_index, self.train_authors, self.train_papers = self.get_train_idx()
        random.shuffle(self.train_index)
        self.real_train_index = self.train_index[:int(len(self.train_index) * train_ratio)]
        self.real_test_index = self.train_index[int(len(self.train_index) * train_ratio):]
        self.total_train_cnt = len(self.train_index)
        self.author_adj_matrix = self.get_author_adj_matrix()
        self.author_author_map = self.get_author_author_map()
        self.paper_adj_matrix = self.get_paper_adj_matrix()
        self.paper_paper_map = self.get_paper_paper_map()
        # self.paper_embeddings = self.get_paper_embeddings()
        self.author_paper_map = self.get_author_paper_map()
        self.bipartite_adj_matrix, self.bipartite_lap_matrix = self.get_bipartite_matrix()

        
        
        with open(self.author_feature_path, 'rb') as f:
            self.author_embeddings = pickle.load(f)

        with open(self.paper_feature_path, 'rb') as f:
            self.paper_embeddings = pickle.load(f)

        self.paper_embeddings = self.paper_embeddings.to(self.device)
        self.paper_embeddings = F.normalize(self.paper_embeddings, 2, 1)
        self.author_embeddings = self.author_embeddings.to(self.device)
        self.paper_paper_nei_embeddings = self.get_paper_paper_nei()


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
            with open(self.author_author_map_path, 'rb') as f:
                author_author_map = pickle.load(f)
            print(f'Load author-author map from {self.author_author_map_path}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(self.author_graph_path, 'r') as f:
                lines = f.readlines()
                author_author_map = {author: set() for author in range(self.author_cnt)}
                for line in lines:
                    for author, coauthor in [line.strip().split(' ')]:
                        author_author_map[int(author)].add(int(coauthor))
                        author_author_map[int(coauthor)].add(int(author))
            print(f'Build author-author map, time cost: {time.time() - t1: .3f}s')
            with open(self.author_author_map_path, 'wb') as f:
                pickle.dump(author_author_map, f)
        return author_author_map

    def get_author_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the coauthor graph.
        Returns:
            The author adjacency matrix.
        """
        try:
            t1 = time.time()
            with open(self.author_adj_path, 'rb') as f:
                author_adj_matrix = pickle.load(f)
            print(f'Load author adjacency matrix from {self.author_adj_path}, time cost: {time.time() - t1: .3f}s')
        except:
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
        author_adj_matrix = author_adj_matrix.to(self.device,dtype=torch.float)
        return author_adj_matrix

    def get_paper_paper_nei(self):
        try: 
            t1 = time.time()
            # with open(self.paper_paper_nei_path, 'rb') as f:
            #     neighbor_embedding = pickle.load(f)
            neighbor_embedding = torch.load(self.paper_paper_nei_path)
            print(f"Load paper-paper map from {self.paper_paper_nei_path}, time cost: {time.time() - t1:.3f}s")
            
        except:
            t1 = time.time()
            
            paper_feature = self.paper_embeddings
            paper_paper_map = self.paper_paper_map

            # max_connection = max(len(list(x)) for i, x in paper_paper_map.items())
            sample_number = 64

            neighbor_embedding = torch.zeros((self.paper_cnt, sample_number + 1, paper_feature.shape[-1]), device=self.device, dtype=paper_feature.dtype)
            for i, x in paper_paper_map.items():
                possible_idx = list(x)
                random.shuffle(possible_idx)
                idx = torch.LongTensor(possible_idx if len(possible_idx) <= sample_number else possible_idx[:sample_number], device=self.device)

                gather_idx = idx.unsqueeze(-1).repeat((1, paper_feature.shape[-1]))

                gathered_embedding = paper_feature.gather(0, gather_idx)
                neighbor_embedding[i] = neighbor_embedding[i].scatter(0, 
                                                                      torch.arange(1, len(idx) + 1, 1, dtype=torch.int64, device=self.device).unsqueeze(-1).repeat((1, paper_feature.shape[-1])), gathered_embedding)
                neighbor_embedding[i, 0] = paper_feature[i]

            print(f'Build paper paper neighborhood, time cost: {time.time() - t1:.3f}')
            torch.save(neighbor_embedding, self.paper_paper_nei_path)
            # with open(self.paper_paper_nei_path, 'wb') as f:
            #     pickle.dump(neighbor_embedding, f)
        
        return neighbor_embedding
                
    
    def get_paper_paper_map(self) -> Dict[int, Set[int]]:
        """Returns the mapping from papers to papers in citation network.
        Returns:
            The mapping from papers to papers.
        Example:
            paper 1 has cited paper 10, 11
            return {1: {10, 11}, 10: {1}, 11: {1}}
        Note: all paper indexes are in range [0, self.paper_cnt)
        """
        try:
            t1 = time.time()
            with open(self.paper_paper_map_path, 'rb') as f:
                paper_paper_map = pickle.load(f)
            print(f'Load paper-paper map from {self.paper_paper_map_path}, time cost: {time.time() - t1: .3f}s')
        except:
            t1 = time.time()
            with open(self.paper_graph_path, 'r') as f:
                lines = f.readlines()
                paper_paper_map = {paper: set() for paper in range(self.paper_cnt)}
                for line in lines:
                    for paper, cited_paper in [line.strip().split(' ')]:
                        paper_paper_map[int(paper)].add(int(cited_paper))
                        paper_paper_map[int(cited_paper)].add(int(paper))
            print(f'Build paper-paper map, time cost: {time.time() - t1: .3f}s')
            with open(self.paper_paper_map_path, 'wb') as f:
                pickle.dump(paper_paper_map, f)
        return paper_paper_map

    def get_paper_adj_matrix(self) -> SparseTensor:
        """Returns the adjacency matrix of the citation network among papers.
        Returns:
            The paper adjacency matrix.
        """
        try:
            t1 = time.time()
            with open(self.paper_adj_path, 'rb') as f:
                paper_adj_matrix = pickle.load(f)
            print(f'Load paper adjacency matrix from {self.paper_adj_path}, time cost: {time.time() - t1: .3f}s')
        except:
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
        paper_adj_matrix = paper_adj_matrix.to(self.device,dtype=torch.float)
        return paper_adj_matrix

    def get_paper_embeddings(self) -> torch.Tensor:
        """Returns the initial embedding of paper from feature file generated by USE
        Returns:
            The USE feature of papers.
        """
        paper_embs = []
        t1 = time.time()
        
        with open(self.paper_feature_path, 'rb') as f:
            paper_embs = pickle.load(f)
        print(f'Load paper embeddings from {self.paper_feature_path}, time cost: {time.time() - t1: .3f}s')
        
        # papar_embs = papar_embs.to(self.device)
        return paper_embs

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
        try:
            t1 = time.time()
            with open(self.train_idx_path, 'rb') as f:
                train_idx = pickle.load(f)
            with open(self.train_authors_path, 'rb') as f:
                train_authors = pickle.load(f)
            with open(self.train_papers_path, 'rb') as f:
                train_papers = pickle.load(f)
            print(f'Load training data from {self.train_idx_path}, time cost: {time.time() - t1: .3f}s')
        except:
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
        self.train_author_cnt = len(train_authors)
        self.train_paper_cnt = len(train_papers)
        return train_idx, train_authors, train_papers

    def get_bipartite_matrix(self) -> Tuple[SparseTensor, SparseTensor]:
        """Returns the adjacency matrix and Laplacian matrix of the bipartite graph.
        Returns:
            The bipartite adjacency matrix and Laplacian matrix.
        """
        bipartite_adj_matrix = []
        bipartite_lap_index = []
        try:
            t1 = time.time()
            with open(self.train_idx_path, 'rb') as f:
                self.train_index = pickle.load(f)
            with open(self.bipartite_adj_path, 'rb') as f:
                bipartite_adj_matrix = pickle.load(f)
            with open(self.bipartite_lap_path, 'rb') as f:
                bipartite_lap_matrix = pickle.load(f)
            print(f'Load train indexed from {self.train_idx_path}')
            print(f'Load bipartite adjacency matrix and Laplacian matrix from {self.bipartite_adj_path} and {self.bipartite_lap_path}, time cost: {time.time() - t1: .3f}s')
        except:
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
            print(f'Build train indexes.')
            print(f'Build bipartite adjacency matrix and Laplacian matrix, time cost: {time.time() - t1: .3f}s')
            with open(self.train_idx_path, 'wb') as f:
                pickle.dump(self.train_index, f)
            with open(self.bipartite_adj_path, 'wb') as f:
                pickle.dump(bipartite_adj_matrix, f)
            with open(self.bipartite_lap_path, 'wb') as f:
                pickle.dump(bipartite_lap_matrix, f)
        bipartite_adj_matrix = bipartite_adj_matrix.to(self.device)
        bipartite_lap_matrix = bipartite_lap_matrix.to(self.device)
        return bipartite_adj_matrix, bipartite_lap_matrix
    
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
        try:
            t1 = time.time()
            with open(self.author_paper_map_path, 'rb') as f:
                author_paper_map = pickle.load(f)
            print(f'Load author-paper map from {self.author_paper_map_path}, time cost: {time.time() - t1: .3f}s')
        except:
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
        return author_paper_map

    def sample(self) -> Tuple[Tensor, Tensor]:
        """
        Sample two random walk path starting from two random nodes(author & paper)
        Returns:
            author_path (Tensor): (random_walk_length, 1)
            paper_path (Tensor): (random_walk_length, 1)
        """
        author_path = self.generate_random_walk_author(self.random_walk_length)
        paper_path = self.generate_random_walk_paper(self.random_walk_length)
        author_path = author_path.to(self.device)
        paper_path = paper_path.to(self.device)
        return author_path, paper_path
    
    def generate_random_walk_author(self, t: int) -> Tensor:
        """Generate the random walk for all authors in coauthor network.

        Args:
            t (int): the length of random walk (t << N).
        Returns:
            random_walk_matrix (Tensor): (self.author_cnt, t)
        """
        random_walk_matrix = torch.zeros(size=(t, 1), dtype=torch.int64)
        start_author = random.choice(range(0, self.author_cnt))
        random_walk_matrix[0, 0] = start_author
        pre = start_author
        for i in range(1, t):
            cur = random.choice(list(self.author_author_map[pre]))
            random_walk_matrix[i, 0] = cur
            pre = cur
        random_walk_matrix = random_walk_matrix.to(self.device)
        return random_walk_matrix

    def generate_random_walk_paper(self, t: int) -> Tensor:
        """Generate the random walk for all papers in citation network.

        Args:
            t (int): the length of random walk (t << M).
        Returns:
            random_walk_matrix (Tensor): (t, 1)
        """
        random_walk_matrix = torch.zeros(size=(t, 1), dtype=torch.int64)
        start_paper = random.choice(range(0, self.paper_cnt))
        random_walk_matrix[0, 0] = start_paper
        pre = start_paper
        for i in range(1, t):
            cur = random.choice(list(self.paper_paper_map[pre]))
            random_walk_matrix[i, 0] = cur
            pre = cur
        random_walk_matrix = random_walk_matrix.to(self.device)
        return random_walk_matrix
    
    def sample_train(self) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
        """ Sample a batch from the train dataset.
        Returns:
            pos_train_index (List[List[int]]): (batch_size // 2, 2)
            neg_train_index (List[List[int]]): (batch_size // 2, 2)
            train_authors (List[int])
            train_papers (List[int])
        """
        pos_train_num = self.batch_size // 2
        neg_train_num = self.batch_size - pos_train_num

        # sample positive
        pos_train_index = random.sample(self.real_train_index, pos_train_num)

        # sample negative
        author_list = list(range(self.n_authors))
        neg_train_authors = random.sample(author_list, neg_train_num)
        neg_train_index = []
        for neg_train_author in neg_train_authors:
            flag = False
            while not flag:
                random_paper = np.random.randint(low=self.author_cnt, high=self.author_cnt + self.n_papers, size=1)[0]
                if random_paper not in self.author_paper_map[neg_train_author]:
                    neg_train_index.append([neg_train_author, random_paper - self.author_cnt])
                    flag = True

        # get authors and papers
        train_authors = list(set([pos[0] for pos in pos_train_index] + [neg[0] for neg in neg_train_index]))
        train_papers = list(set([pos[1] for pos in pos_train_index] + [neg[1] for neg in neg_train_index]))

        return pos_train_index, neg_train_index, train_authors, train_papers
    
    def sample_test(self) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
        """ Sample a batch from the test dataset.
        Returns:
            pos_test_index (List[List[int]]): (batch_size // 2, 2)
            neg_test_index (List[List[int]]): (batch_size // 2, 2)
            test_authors (List[int])
            test_papers (List[int])
        """
        pos_test_num = self.batch_size // 2
        neg_test_num = self.batch_size - pos_test_num

        # sample positive
        pos_test_index = random.sample(self.real_test_index, pos_test_num)

        # sample negative
        author_list = list(range(self.n_authors))
        neg_test_authors = random.sample(author_list, neg_test_num)
        neg_test_index = []
        for neg_test_author in neg_test_authors:
            flag = False
            while not flag:
                random_paper = np.random.randint(low=self.author_cnt, high=self.author_cnt + self.n_papers, size=1)[0]
                if random_paper not in self.author_paper_map[neg_test_author]:
                    neg_test_index.append([neg_test_author, random_paper - self.author_cnt])
                    flag = True

        # get authors and papers
        test_authors = list(set([pos[0] for pos in pos_test_index] + [neg[0] for neg in neg_test_index]))
        test_papers = list(set([pos[1] for pos in pos_test_index] + [neg[1] for neg in neg_test_index]))

        return pos_test_index, neg_test_index, test_authors, test_papers

    def get_train_test_indexes(self) -> Tuple[List[List[int]], List[List[int]],List[List[int]], List[List[int]], List[int], List[int], List[int], List[int]]:
        assert self.train_index != None
        t1 = time.time()
        random.shuffle(self.train_index)
        # pos_train_num = int(self.total_train_cnt * train_ratio)
        # pos_test_num = self.total_train_cnt - pos_train_num
        pos_train_num = self.batch_size // 2
        pos_test_num = self.batch_size // 2

        # postive train/test
        real_train_pos_index = self.train_index[:pos_train_num]
        real_test_pos_index = self.train_index[-pos_test_num:]
        t2 = time.time()
        print(f'Get positive train/test indexes, time cost: {t2 - t1: .3f}s')

        # negative train/test
        real_train_neg_index = []
        real_test_neg_index = []
        author_list = list(range(self.n_authors))
        neg_train_authors = [random.choice(author_list) for _ in range(pos_train_num)]
        neg_test_authors = [random.choice(author_list) for _ in range(pos_test_num)]
        for neg_train_author in neg_train_authors:
            flag = False
            while not flag:
                random_paper = np.random.randint(low=self.author_cnt, high=self.author_cnt + self.n_papers, size=1)[0]
                if random_paper not in self.author_paper_map[neg_train_author]:
                    real_train_neg_index.append([neg_train_author, random_paper - self.author_cnt])
                    flag = True
        for neg_test_author in neg_test_authors:
            flag = False
            while not flag:
                random_paper = np.random.randint(low=self.author_cnt, high=self.author_cnt + self.n_papers, size=1)[0]
                if random_paper not in self.author_paper_map[neg_test_author]:
                    real_test_neg_index.append([neg_test_author, random_paper - self.author_cnt])
                    flag = True
        t3 = time.time()
        print(f'Get negative train/test indexes, time cost: {t3 - t2: .3f}s')

        # get the set of authors and papers in train dataset and test dataset for regularization loss
        real_train_authors = list(set([pos[0] for pos in real_train_pos_index] + [neg[0] for neg in real_train_neg_index]))
        real_train_papers = list(set([pos[1] for pos in real_train_pos_index] + [neg[1] for neg in real_train_neg_index]))
        real_test_authors = list(set([pos[0] for pos in real_test_pos_index] + [neg[0] for neg in real_test_neg_index]))
        real_test_papers = list(set([pos[1] for pos in real_test_pos_index] + [neg[1] for neg in real_test_neg_index]))
        t4 = time.time()
        print(f'Get train/test authors and papers, time cost: {t4 - t3: .3f}s')

        return real_train_pos_index, real_train_neg_index, real_test_pos_index, real_test_neg_index, real_train_authors, real_train_papers, real_test_authors, real_test_papers

if __name__ == '__main__':
    data_generator = Data(batch_size=1024, random_walk_length=16, device='cpu', path='data')
    # print(data_generator.author_author_map[0])
    # print(data_generator.paper_paper_map[0])
    # print('--author--')
    # path = data_generator.sample()[0]
    # for i in path:
    #     print(i, data_generator.author_author_map[int(i)])
    # print('--paper--')
    # path = data_generator.sample()[1]
    # for i in path:
    #     print(i, data_generator.paper_paper_map[int(i)])
    # print(data_generator.train_index[:10])
    # print(data_generator.author_adj_matrix[0])
    # print(data_generator.paper_adj_matrix[0])
    # print(data_generator.bipartite_adj_matrix[0])
    # print(len(data_generator.train_index))
    # print(data_generator.train_index[:10])
    # print(len(data_generator.train_authors))
    # print(data_generator.train_authors[:10])
    # print(len(data_generator.train_papers))
    # print(data_generator.train_papers[:10])
    # a, b, c, d = data_generator.sample_train()
    # e, f, g, h = data_generator.sample_test()
    # print(a[:10])
    # print(b[:10])
    # print(c[:10])
    # print(d[:10])
    # print(list(e)[:10])
    # print(list(f)[:10])
    # print(list(g)[:10])
    # print(list(h)[:10])
    # print('-------------')
    # print(len(a))
    # print(len(b))
    # print(len(c))
    # print(len(d))
    # print(len(e))
    # print(len(f))
    # print(len(g))
    # print(len(h))
    

