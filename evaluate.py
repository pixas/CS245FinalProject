import argparse
import os

import numpy as np
from models.General import General
import time
import torch

from tqdm import tqdm
from utils.dataset import AcademicDataset
from typing import List

TRAIN_FILE_TXT = 'data/bipartite_train.txt'
TEST_FILE_TXT = 'data/bipartite_test_ann.txt'
# TODO: load from file
def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument("path", type=str, default='checkpoints', help='checkpoint to reuse for evaluation')
    parser.add_argument('--output_dir', type=str, default='data', help='directory to save output csv files')

    return parser.parse_args()

args = parse_args()
model_parameter = torch.load(args.path)
train_args = model_parameter['args']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = AcademicDataset(batch_size=args.batch_size, random_walk_length=args.rw_length, device=device, path=args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
pretrained_author_embedding = torch.arange(0, data_generator.author_cnt, 1, device=device)
pretrained_paper_embedding = data_generator.get_paper_embeddings()
paper_connect_author = data_generator.get_paper_connect_author()
paper_mask = data_generator.paper_mask
paper_paper_map = data_generator._paper_paper_map
paper_mask = torch.tensor(paper_mask,dtype=torch.float32,device=device)



@torch.no_grad()
def evaluate_test_ann(model: General, test_file: str, output_dir: str):
    n_test_batch = len(data_generator.real_test_index) // (data_generator.batch_size // 2) + 1
    output_file_name = "13_ShuyangJiang.csv"
    test_array: list = np.loadtxt(test_file, dtype=int, delimiter=' ').tolist()
    model.eval()
    # author_embedding = pretrained_author_embedding
    # paper_embedding = pretrained_paper_embedding
    with tqdm(total=n_test_batch) as t:
        t.set_description(f"Evaluation")

        for batch_idx in range(1, n_test_batch + 1):

            test_pos_index, test_neg_index, test_authors, test_papers = data_generator.sample_test()
            # paper_neighbor_embedding = data_generator.get_batch_paper_neighbor(pretrained_paper_embedding, test_papers)
            paper_neighbor_embedding= []
            author_embedding, paper_embedding, interact_prob = model(
                pretrained_author_embedding, 
                pretrained_paper_embedding,
                test_authors,
                test_papers,
                paper_paper_map,
                paper_mask
            )

            t.update(1)
    f = open(os.path.join(output_dir, output_file_name), 'w')
    f.write("Index,Probability\n")
    with tqdm(total=len(test_array)) as t:
        t.set_description("Evaluating:")
        for idx, (author, paper) in enumerate(test_array):
            result = torch.sum(author_embedding[author] * paper_embedding[paper])
            prob = torch.sigmoid(result).item()
            f.write("{},{}\n".format(idx, prob))
            t.update(1)
    f.close()


        
    


if __name__ == '__main__':
    
    model = General(
        Pa_layers=train_args.pa_layers,
        Au_layers=train_args.au_layers,
        paper_adj=data_generator.get_paper_adj_matrix(),
        author_adj=data_generator.get_author_adj_matrix(),
        Paperdropout=train_args.gnn_dropout,
        Authordropout=train_args.gnn_dropout,
        n_authors=data_generator.author_cnt,
        n_papers=data_generator.paper_cnt,
        num_layers=train_args.NGCF_layers,
        NGCFembed_dim=train_args.embed_dim,
        dropoutNGCF=train_args.ngcf_dropout,
        paper_dim=train_args.embed_dim,
        author_dim=train_args.embed_dim,
        layer_size_list=train_args.layer_size_list,
        args=train_args
    )
    model.to(device)
    model.load_state_dict(model_parameter['model_state'])
    evaluate_test_ann(model, TEST_FILE_TXT, args.output_dir)
    