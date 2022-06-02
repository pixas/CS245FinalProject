import argparse
import os
from typing import List
from models.General import General
import time
import torch
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils.dataset import AcademicDataset
from functools import cmp_to_key
from torch.utils.tensorboard import SummaryWriter  

TRAIN_FILE_TXT = 'data/bipartite_train.txt'
def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument('save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='checkpoints', help='directory to save tensorboard info')
    parser.add_argument('--exp_name', type=str, default='exp')
    
    # Optimization Parameters
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
    parser.add_argument('--rw_length', type=int, default=512, help='random walk length')
    parser.add_argument('--only_feature', action='store_true', default=True)
    parser.add_argument('--layer_size_list', type=List[int], default=[512, 768, 1024], help='increase of receptive field')
    
    
    # GAT Parameters
    parser.add_argument('--num_heads', type=int, default=8, help='multihead attention heads')
    parser.add_argument('--gat_layers', type=int, default=6, help='GAT stack layers')
    
    # GCN Parameters
    parser.add_argument('--pa_layers', type=int, default=2, help='paper GNN layers')
    parser.add_argument('--au_layers', type=int, default=2, help='author GNN layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.2, help='GNN layer dropout rate')
    
    
    parser.add_argument('--decay', type=float, default=0.1, help='regularizer term coefficient')
    
    # dataset Parameters
    parser.add_argument('--sample_number', type=int, default=64, help='gat sample number for attention')
    parser.add_argument('--datapath', type=str, default='data', help='data path')
    return parser.parse_args()

# TODO: load from file
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensorbaord_dir = os.path.join(args.log_dir, args.exp_name)

data_generator = AcademicDataset(batch_size=args.batch_size, random_walk_length=args.rw_length, device=device, path=args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
init_author_embedding = torch.arange(0, data_generator.author_cnt, 1, device=device)
init_paper_embedding = torch.arange(0, data_generator.paper_cnt, 1, device=device)
paper_feature = data_generator.get_paper_embeddings()
adj_matrix, lap_matrix = data_generator.get_bipartite_matrix()

def get_loss(author_embedding, paper_embedding, decay, pos_index, neg_index, authors, papers):
    # interact prob
    interact_prob = (author_embedding * paper_embedding).sum(-1)  # (B, )
    interact_prob = torch.sigmoid(interact_prob)
    batch_size = interact_prob.shape[0]

    assert len(pos_index) + len(neg_index) == batch_size
    # score_matrix = torch.matmul(author_embedding, paper_embedding.transpose(0, 1))
    

    pos_scores = interact_prob[:batch_size // 2]
    neg_scores = interact_prob[batch_size // 2:]
    target = torch.ones_like(interact_prob).to(interact_prob)
    target[batch_size // 2:] = 0.
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # bce_loss = loss_fn(interact_prob, target)
    bce_loss = (torch.sum(1-pos_scores) + torch.sum(neg_scores)) / (len(pos_index) + len(neg_index))

    
    # mf_loss = torch.sum(1 - pos_scores + neg_scores) / (len(pos_index) + len(neg_index))

    regularizer = (torch.norm(author_embedding) ** 2 + torch.norm(paper_embedding) ** 2) / 2
    emb_loss = decay * regularizer / (len(authors) + len(papers))
    
    # pred_pos = torch.sum(score_matrix >= 0)
    pos_samples = pos_scores >= 0.5
    neg_samples = neg_scores >= 0.5
    true_pos = torch.sum(pos_samples)
    precision = torch.sum(pos_samples) / (torch.sum(pos_samples) + torch.sum(neg_samples))
    recall = true_pos / len(pos_index)

    return bce_loss + emb_loss, bce_loss, emb_loss, precision, recall

def save_checkpoint(model: General, args: argparse.ArgumentParser, save_metric: float, epoch: int):
    keep_last_epochs = args.keep_last_epochs
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_ckpts = os.listdir(save_dir)
    all_ckpts = list(filter(lambda x: x.startswith('checkpoint'), all_ckpts))
    cur_save_info = {
            'model_state': model.state_dict(),
            'metric': save_metric,
            'epoch': epoch,
            'args': args
        }
    if len(all_ckpts) == 0:
        torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_last.pt"))
        torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_best.pt"))
        torch.save(cur_save_info, os.path.join(save_dir, "checkpoint{}.pt".format(epoch)))
    else:
        if len(all_ckpts) - 2 >= keep_last_epochs:

            all_ckpts.remove('checkpoint_best.pt')
            all_ckpts.remove('checkpoint_last.pt')
            all_ckpts.sort(key=cmp_to_key(lambda x, y: int(x[10:-3]) - int(y[10:-3])))
            os.system("rm -rf {}".format(os.path.join(save_dir, all_ckpts[0])))
            torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_last.pt"))
            torch.save(cur_save_info, os.path.join(save_dir, "checkpoint{}.pt".format(epoch)))
            best_model_info = torch.load(os.path.join(save_dir, 'checkpoint_best.pt'))
            metric = best_model_info['metric']
            if save_metric > metric:
                torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_best.pt"))
        else:
            torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_last.pt"))
            torch.save(cur_save_info, os.path.join(save_dir, "checkpoint{}.pt".format(epoch)))
            best_model_info = torch.load(os.path.join(save_dir, 'checkpoint_best.pt'))
            metric = best_model_info['metric']
            if save_metric > metric:
                torch.save(cur_save_info, os.path.join(save_dir, "checkpoint_best.pt"))
            
    return

@torch.no_grad()
def test_one_epoch(model: General, args: argparse.ArgumentParser, epoch_idx: int):
    n_test_batch = len(data_generator.real_test_index) // (data_generator.batch_size // 2) + 1
    model.eval()

    # paper_embedding = pretrained_paper_embedding
    with tqdm(total=n_test_batch) as t:
        t.set_description(f"Test Epoch {epoch_idx}")
        epoch_loss, epoch_bce_loss, epoch_emb_loss = 0, 0, 0
        epoch_total_precision, epoch_total_recall = 0, 0
        for batch_idx in range(1, n_test_batch + 1):

            test_pos_index, test_neg_index, test_authors, test_papers = data_generator.sample_test()
            # paper_neighbor_embedding = data_generator.get_batch_paper_neighbor(pretrained_paper_embedding, test_papers)

            author_embedding, paper_embedding = model(
                init_author_embedding, 
                init_paper_embedding,
                paper_feature,
                test_papers,
                test_authors
            )

            test_loss, test_bce_loss, test_emb_loss, test_precision, test_recall = get_loss(author_embedding, paper_embedding, args.decay, test_pos_index, test_neg_index, test_authors, test_papers)
            
            epoch_loss += test_loss
            epoch_bce_loss += test_bce_loss
            epoch_emb_loss += test_emb_loss
            epoch_total_precision += test_precision
            epoch_total_recall += test_recall
            
            t.set_postfix({"loss": f"{test_loss:.4f}", 'bce_loss': f"{test_bce_loss:.4f}", 'precision': f"{test_precision:.4f}", 'recall': f"{test_recall:.4f}"})
            t.update(1)
    
    test_loss = epoch_loss / n_test_batch
    test_bce_loss = epoch_bce_loss / n_test_batch
    test_total_precision = epoch_total_precision / n_test_batch
    test_total_recall = epoch_total_recall / n_test_batch
    return test_loss, test_bce_loss, test_total_precision, test_total_recall


def train(model: General, optimizer, args):
    train_writer = SummaryWriter(os.path.join(tensorbaord_dir, 'train'))
    test_writer = SummaryWriter(os.path.join(tensorbaord_dir, 'test'))
    epoch = args.epoch
    begin_epoch = 1
    if os.path.exists(args.save_dir):
        ckpt_dir = os.listdir(args.save_dir)
        if 'checkpoint_last.pt' in ckpt_dir:
            last_model_dict = torch.load(os.path.join(args.save_dir, 'checkpoint_last.pt'))
            parameter_dict = last_model_dict['model_state']
            begin_epoch = last_model_dict['epoch'] + 1
            model.load_state_dict(parameter_dict)
    else:
        begin_epoch = 1

    # paper_embedding = pretrained_paper_embedding
    
    for epoch_idx in range(begin_epoch, epoch + 1):
        n_train_batch = len(data_generator.real_train_index) // (data_generator.batch_size // 2) + 1
        model.train()
        with tqdm(total=n_train_batch) as t:
            t.set_description(f"Train Epoch {epoch_idx}")
            epoch_loss, epoch_bce_loss, epoch_emb_loss = 0, 0, 0
            epoch_total_precision, epoch_total_recall = 0, 0

            for batch_idx in range(1, n_train_batch + 1):
                # author_path, paper_path = data_generator.sample()

                train_pos_index, train_neg_index, train_authors, train_papers = data_generator.sample_train()
                # paper_neighbor_embedding = data_generator.get_batch_paper_neighbor(pretrained_paper_embedding, train_papers)
                author_embedding, paper_embedding = model(
                    init_author_embedding, 
                    init_paper_embedding,
                    paper_feature,
                    train_papers,
                    train_authors
                )
                
                # train_pos_index, train_neg_index, test_pos_index, test_neg_index, train_authors, train_papers, test_authors, test_papers = data_generator.get_train_test_indexes()
                loss, bce_loss, emb_loss, precision, recall = get_loss(author_embedding, paper_embedding, args.decay, train_pos_index, train_neg_index, train_authors, train_papers)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                epoch_bce_loss += bce_loss
                epoch_emb_loss += emb_loss
                epoch_total_precision += precision
                epoch_total_recall += recall
                t.set_postfix({"loss": f"{loss:.4f}", 'bce_loss': f"{bce_loss:.4f}", 'emb_loss': f"{emb_loss:.4f}", 'precision': f"{precision:.4f}", 'recall': f"{recall:.4f}"})
                t.update(1)
        print(f'Train Epoch {epoch_idx:.4f} Loss: {epoch_loss / n_train_batch:.4f} MF Loss: {epoch_bce_loss / n_train_batch:.4f} Emb Loss: {epoch_emb_loss / n_train_batch:.4f} Precision: {epoch_total_precision / n_train_batch:.4f} Recall: {epoch_total_recall / n_train_batch:.4f}')
        
        train_writer.add_scalar('Train/total loss', epoch_loss / n_train_batch, epoch)
        train_writer.add_scalar('Train/bce loss', epoch_bce_loss / n_train_batch, epoch)
        train_writer.add_scalar('Train/embedding loss', epoch_emb_loss / n_train_batch, epoch)
        test_loss, test_bce_loss, test_total_precision, test_total_recall = test_one_epoch(model, args, epoch_idx)
        print(f'Test Epoch {epoch_idx:.4f} Loss: {test_loss:.4f} BCE Loss: {test_bce_loss:.4f} Precision: {test_total_precision:.4f} Recall: {test_total_recall:.4f}')
        test_writer.add_scalar('Valid/total loss', test_loss, epoch)
        test_writer.add_scalar('Valid/mf loss', test_bce_loss, epoch)
        test_writer.add_scalar('Valid/precision', test_total_precision, epoch)
        test_writer.add_scalar('Valid/recall', test_total_recall, epoch)
        save_checkpoint(model, args, test_total_recall, epoch_idx)
        print("*" * 100)
        # np.save(os.path.join(args.save_dir, 'interact_prob.npy'), interact_prob.detach().cpu().numpy())




if __name__ == '__main__':
    
    model = General(
        Pa_layers=args.pa_layers,
        Au_layers=args.au_layers,
        paper_adj=data_generator.get_paper_adj_matrix(),
        author_adj=data_generator.get_author_adj_matrix(),
        Paperdropout=args.gnn_dropout,
        Authordropout=args.gnn_dropout,
        n_authors=data_generator.author_cnt,
        n_papers=data_generator.paper_cnt,
        num_layers=args.NGCF_layers,
        NGCFembed_dim=args.embed_dim,
        dropoutNGCF=args.ngcf_dropout,
        paper_dim=args.embed_dim,
        author_dim=args.embed_dim,
        layer_size_list=args.layer_size_list,
        args=args,
        norm_adj=adj_matrix,
        only_feature=args.only_feature
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, args)
