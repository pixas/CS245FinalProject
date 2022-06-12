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

# TODO: load from file
def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument("path", type=str, default='checkpoints', help='checkpoint to reuse for evaluation')
    parser.add_argument('--output_name', type=str, default='test_table')
    parser.add_argument('--output_dir', type=str, default='data', help='directory to save output csv files')

    return parser.parse_args()




args = parse_args()
print(args)
model_parameter = torch.load(args.path)
train_args = model_parameter['args']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = AcademicDataset(batch_size=train_args.batch_size,device=device, path=train_args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
init_author_embedding = torch.arange(0, data_generator.author_cnt, 1, device=device)
init_paper_embedding = torch.arange(0, data_generator.paper_cnt, 1, device=device)
paper_feature = data_generator.get_paper_embeddings()
paper_paper_map, paper_padding_mask = data_generator.get_paper_paper_map()
adj_matrix, lap_matrix = data_generator.get_bipartite_matrix()
TRAIN_FILE_TXT = f'{train_args.datapath}/bipartite_train.txt'
TEST_FILE_TXT = f'{train_args.datapath}/bipartite_test_ann.txt'
output_file_name = args.output_name

@torch.no_grad()
def evaluate_test_ann(model: General, test_file: str, output_dir: str):
    
    test_array: list = np.loadtxt(test_file, dtype=int, delimiter=' ')
    model.eval()
    # author_embedding = pretrained_author_embedding
    # paper_embedding = pretrained_paper_embedding
    test_authors = test_array[:, 0].tolist()
    test_papers = test_array[:, 1].tolist()
    batch_size = train_args.batch_size
    iter_times = len(test_papers) // batch_size
    final_author_embeddings = []
    final_paper_embeddings = []
    for i in range(iter_times + 1):
        batch_test_authors = test_authors[i * batch_size: (i + 1) * batch_size]
        batch_test_papers = test_papers[i * batch_size: (i + 1) * batch_size]
        if not batch_test_papers:
            break
        author_embedding, paper_embedding, _ = model(
            init_author_embedding,
            init_paper_embedding,
            paper_feature,
            batch_test_papers,
            batch_test_authors,
            paper_paper_map,
            paper_padding_mask
        )

        final_author_embeddings.append(author_embedding)
        final_paper_embeddings.append(paper_embedding)
    

        
    test_author_embedding = torch.cat(final_author_embeddings, 0)
    test_paper_embedding = torch.cat(final_paper_embeddings, 0)
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
        num_layers=train_args.lightgcn_layers,
        lightgcn_dropout=train_args.lightgcn_dropout,
        paper_dim=train_args.embed_dim,
        author_dim=train_args.embed_dim,
        layer_size_list=train_args.layer_size_list,
        only_feature=train_args.only_feature,
        use_lightgcn=train_args.use_lightgcn if train_args.use_lightgcn else True,
        args=train_args,
        norm_adj=lap_matrix
    )
    model.to(device)
    model.load_state_dict(model_parameter['model_state'])
    evaluate_test_ann(model, TEST_FILE_TXT, args.output_dir)
    