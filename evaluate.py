import argparse
import os

import numpy as np
from models.General import General
import time
import torch
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm
from utils.data import Data


TRAIN_FILE_TXT = 'data/bipartite_train.txt'
TEST_FILE_TXT = 'data/bipartite_test_ann.txt'
# TODO: load from file
def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument('save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--module_type', nargs='?', default='LSTM', help='Module in coauthor and citation network')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--keep_last_epochs', type=int, default=5, help='only keep last epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='batch samples for positive and negative samples')
    parser.add_argument('--embed_dim', type=int, default=512, help='embedding dimension')
    parser.add_argument('--rw_stack_layers', type=int, default=2, help='random walk module stack layers')
    parser.add_argument('--rw_dropout', type=float, default=0.3, help='random walk dropout rate')
    parser.add_argument('--NGCF_layers', type=int, default=3, help='ngcf layers')
    parser.add_argument('--ngcf_dropout', type=float, default=0.3, help='ngcf dropout rate')
    
    return parser.parse_args()

args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = Data(batch_size=args.batch_size, random_walk_length=16,device=device)
pretrained_author_embedding = data_generator.author_embeddings
pretrained_paper_embedding = data_generator.paper_embeddings



@torch.no_grad()
def evaluate_test_ann(model: General, test_file: str, output_dir: str):
    output_file_name = "13_ShuyangJiang.csv"
    test_array: list = np.loadtxt(test_file, dtype=int, delimiter=' ').tolist()
    model.eval()
    author_path, paper_path = data_generator.sample()
    author_embedding, paper_embedding = model(
        pretrained_author_embedding, 
        pretrained_paper_embedding,
        author_path,
        paper_path
    )
    author_embedding = F.normalize(author_embedding, 2, 1)
    paper_embedding = F.normalize(paper_embedding, 2, 1)
    f = open(os.path.join(output_dir, output_file_name), 'w')
    f.write("Index,Probability\n")
    with tqdm(total=len(test_array)) as t:
        t.set_description("Evaluating:")
        for idx, (author, paper) in enumerate(test_array):
            result = torch.sum(author_embedding[author] * paper_embedding[paper])
            prob = (1 + result.item()) / 2.0
            f.write("{},{}\n".format(idx, prob))
            t.update(1)
    f.close()
        
        
    


if __name__ == '__main__':
    
    model = General(
        RWembed_dim=args.embed_dim,
        stack_layers=args.rw_stack_layers,
        dropoutRW=args.rw_dropout,
        n_authors=data_generator.n_authors,
        n_papers=data_generator.n_papers,
        num_layers=args.NGCF_layers,
        NGCFembed_dim=args.embed_dim,
        dropoutNGCF=args.ngcf_dropout,
        paper_dim=args.embed_dim,
        author_dim=args.embed_dim,
        norm_adj=data_generator.bipartite_lap_matrix,
        n_fold=4,
        args=args
    )
    model.to(device)
    model_parameter = torch.load(args.path)
    model.load_state_dict(model_parameter['model_state'])
    evaluate_test_ann(model, TEST_FILE_TXT, args.output_dir)
    