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

data_generator = AcademicDataset(batch_size=train_args.batch_size, random_walk_length=train_args.rw_length,device=device, path=train_args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
init_author_embedding = torch.arange(0, data_generator.author_cnt, 1, device=device)
init_paper_embedding = torch.arange(0, data_generator.paper_cnt, 1, device=device)
paper_feature = data_generator.get_paper_embeddings()




@torch.no_grad()
def evaluate_test_ann(model: General, test_file: str, output_dir: str):
    n_test_batch = len(data_generator.real_test_index) // (data_generator.batch_size // 2) + 1
    output_file_name = "13_ShuyangJiang.csv"
    test_array: list = np.loadtxt(test_file, dtype=int, delimiter=' ')
    model.eval()
    # author_embedding = pretrained_author_embedding
    # paper_embedding = pretrained_paper_embedding
    test_authors = test_array[:, 0].tolist()
    test_papers = test_array[:, 1].tolist()
    author_embedding, paper_embedding, _ = model(
        init_author_embedding,
        init_paper_embedding,
        paper_feature,
        [],
        test_authors,
        test_papers
    )
    test_author_embedding = author_embedding[test_authors]
    test_paper_embedding = paper_embedding[test_papers]
    predicted_prob = torch.sigmoid((test_author_embedding * test_paper_embedding).sum(1))
    predicted_prob = predicted_prob.detach().cpu().numpy()
    
    f = open(os.path.join(output_dir, output_file_name), 'w')
    f.write("Index,Probability\n")
    with tqdm(total=len(test_array)) as t:
        t.set_description("Evaluating:")
        for idx, (author, paper) in enumerate(test_array):
            prob = predicted_prob[idx]
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
        only_feature=train_args.only_feature,
        args=train_args
    )
    model.to(device)
    model.load_state_dict(model_parameter['model_state'])
    evaluate_test_ann(model, TEST_FILE_TXT, args.output_dir)
    