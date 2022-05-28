import argparse
import os

import numpy as np
from models.General import General
import time
import torch
import torch.nn.functional as F 
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


def get_loss(author_embedding, paper_embedding, interact_prob, decay, pos_index, neg_index, authors, papers):
    author_embeddings = author_embedding[authors]
    paper_embeddings = paper_embedding[papers]
    author_embedding = F.normalize(author_embedding, p=2, dim=1)
    paper_embedding = F.normalize(paper_embedding, p=2, dim=1)
    # score_matrix = torch.matmul(author_embedding, paper_embedding.transpose(0, 1))
    
    fetch_pos_index = list(zip(*pos_index))
    fetch_neg_index = list(zip(*neg_index))
    pos_scores = interact_prob[fetch_pos_index[0], fetch_pos_index[1]]
    neg_scores = interact_prob[fetch_neg_index[0], fetch_neg_index[1]]

    mf_loss = (torch.sum(1-pos_scores) + torch.sum(neg_scores)) / (len(pos_index) + len(neg_index))
    # mf_loss = F.nll_loss(pos_scores, torch.ones((pos_scores.shape[0]), device=interact_prob.device)) + \
    #     F.nll_loss(neg_scores, torch.zeros((pos_scores.shape[0]), device=interact_prob.device))
    
    # mf_loss = torch.sum(1 - pos_scores + neg_scores) / (len(pos_index) + len(neg_index))

    regularizer = (torch.norm(author_embeddings) ** 2 + torch.norm(paper_embeddings) ** 2) / 2
    emb_loss = decay * regularizer / (len(authors) + len(papers))
    
    # pred_pos = torch.sum(score_matrix >= 0)
    pos_samples = pos_scores >= 0.5
    neg_samples = neg_scores >= 0.5
    true_pos = torch.sum(pos_samples)
    precision = torch.sum(pos_samples) / (torch.sum(pos_samples) + torch.sum(neg_samples))
    recall = true_pos / len(pos_index)

    return mf_loss + emb_loss, mf_loss, emb_loss, precision, recall

args = parse_args()
print(args)
model_parameter = torch.load(args.path)
train_args = model_parameter['args']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = AcademicDataset(batch_size=train_args.batch_size, random_walk_length=train_args.rw_length,device=device, path=train_args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
init_author_embedding = torch.arange(0, data_generator.author_cnt, 1, device=device)
init_paper_embedding = torch.arange(0, data_generator.paper_cnt, 1, device=device)
paper_feature = data_generator.get_paper_embeddings()
paper_paper_map, paper_padding_mask = data_generator.get_paper_paper_map()



@torch.no_grad()
def evaluate_test_ann(model: General, test_file: str, output_dir: str):
    output_file_name = "13_ShuyangJiang.csv"
    test_array: list = np.loadtxt(test_file, dtype=int, delimiter=' ')
    model.eval()
    # author_embedding = pretrained_author_embedding
    # paper_embedding = pretrained_paper_embedding
    test_authors = test_array[:, 0].tolist()
    test_papers = test_array[:, 1].tolist()
    batch_size = train_args.batch_size
    iter_times = len(test_papers) // batch_size
    for i in range(iter_times):
        batch_test_authors = test_authors[iter_times * batch_size: (iter_times + 1) * batch_size]
        batch_test_papers = test_papers[iter_times * batch_size: (iter_times + 1) * batch_size]
        if not batch_test_papers:
            break
        author_embedding, paper_embedding, _ = model(
            init_author_embedding,
            init_paper_embedding,
            paper_feature,
            [],
            test_authors,
            test_papers,
            paper_paper_map,
            paper_padding_mask
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
    