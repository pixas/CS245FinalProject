import argparse
from models.General import General
import time
import torch
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm
from utils.data import Data


TRAIN_FILE_TXT = 'data/bipartite_train.txt'

# TODO: load from file
data_generator = Data(random_walk_length=16)
pretrained_author_embedding = torch.zeros((data_generator.n_authors, 128), dtype=torch.float)
pretrained_paper_embedding = data_generator.paper_embeddings

def loss(author_embedding, paper_embedding, decay):
    author_embedding = F.normalize(author_embedding, p=2, dim=1)
    paper_embedding = F.normalize(paper_embedding, p=2, dim=1)
    score_matrix = torch.matmul(author_embedding, paper_embedding.transpose(0, 1))
    train_scores = score_matrix[list(zip(*data_generator.train_idx))]
    mf_loss = torch.sum(1 - train_scores)

    train_users = author_embedding[data_generator.train_authors]
    train_papers = paper_embedding[data_generator.train_papers]
    regularizer = (torch.norm(train_users) ** 2 + torch.norm(train_papers) ** 2) / 2
    emb_loss = decay * regularizer / (data_generator.train_author_cnt + data_generator.train_paper_cnt)
    
    pred_pos = torch.sum(score_matrix >= 0.5)
    true_pos = torch.sum(train_scores >= 0.5)
    precision = true_pos / pred_pos
    recall = true_pos / len(data_generator.train_idx)

    return mf_loss + emb_loss, mf_loss, emb_loss, precision, recall

def train(model, optimizer, epoch):
    for epoch_idx in tqdm(range(epoch)):
        t1 = time.time()
        author_path, paper_path = data_generator.sample()
        author_embedding, paper_embedding = model(
            pretrained_author_embedding, 
            pretrained_paper_embedding,
            author_path,
            paper_path
        )
        loss, mf_loss, emb_loss, precision, recall = loss(author_embedding, paper_embedding, 0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t2 = time.time()
        print(f'epoch: {epoch_idx:>02d}, loss: {loss:.4f}, mf_loss: {mf_loss:.4f}, emb_loss: {emb_loss:.4f}, time: {t2 - t1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')


def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument('--module_type', nargs='?', default='LSTM', help='Module in coauthor and citation network')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = General(
        RWembed_dim=128,
        stack_layers=2,
        dropoutRW=0.3,
        n_authors=data_generator.n_authors,
        n_papers=data_generator.n_papers,
        num_layers=2,
        NGCFembed_dim=128,
        dropoutNGCF=0.3,
        paper_dim=128,
        author_dim=128,
        norm_adj=data_generator.bipartite_lap_matrix,
        n_fold=4,
        args=args
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, epoch=10)