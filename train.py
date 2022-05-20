import argparse
import os
from typing import List
from models.General import General
import time
import torch
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm
from utils.data import Data


TRAIN_FILE_TXT = 'data/bipartite_train.txt'
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
    parser.add_argument('--rw_length', type=int, default=512, help='random walk length')
    parser.add_argument('--layer_size_list', type=List[int], default=[512, 768, 1024], help='increase of receptive field')
    parser.add_argument('--pa_layers', type=int, default=2, help='paper GNN layers')
    parser.add_argument('--au_layers', type=int, default=2, help='author GNN layers')
    parser.add_argument('--decay', type=float, default=0.1, help='regularizer term coefficient')
    parser.add_argument('--gnn_dropout', type=float, default=0.2, help='GNN layer dropout rate')
    parser.add_argument('--datapath', type=str, default='data', help='data path')
    return parser.parse_args()

# TODO: load from file
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = Data(batch_size=args.batch_size, random_walk_length=args.rw_length, device=device, path=args.datapath)
# pretrained_author_embedding = data_generator.author_embeddings
pretrained_author_embedding = torch.arange(0, data_generator.n_authors, 1, device=device)
pretrained_paper_embedding = F.normalize(data_generator.paper_embeddings, p=2, dim=1)


def get_loss(author_embedding, paper_embedding, decay, pos_index, neg_index, authors, papers):
    author_embeddings = author_embedding[authors]
    paper_embeddings = paper_embedding[papers]
    # author_embedding = F.normalize(author_embedding, p=2, dim=1)
    # paper_embedding = F.normalize(paper_embedding, p=2, dim=1)
    
    score_matrix = torch.matmul(author_embedding, paper_embedding.transpose(0, 1))
    score_matrix = F.sigmoid(score_matrix)
    pos_scores = score_matrix[list(zip(*pos_index))]
    neg_scores = score_matrix[list(zip(*neg_index))]
    with open('data/log.txt', 'w') as f:
	
        f.write(str(score_matrix[:10]) + '\n')
        f.write(str(pos_scores[:10]) + '\n')
        f.write(str(neg_scores[:10]) + '\n')
        f.write('\n')
    # mf_loss = torch.sum(1 - pos_scores + neg_scores) / (len(pos_index) + len(neg_index))
    mf_loss = - (torch.sum(torch.log(pos_scores)) + torch.sum(torch.log(1 - neg_scores))) / (len(pos_index) + len(neg_index))

    regularizer = (torch.norm(author_embeddings) ** 2 + torch.norm(paper_embeddings) ** 2) / 2
    emb_loss = decay * regularizer / (len(authors) + len(papers))
    
    # pred_pos = torch.sum(score_matrix >= 0)
    true_pos = torch.sum(pos_scores >= 0)
    precision = torch.sum(pos_scores >= 0) / (torch.sum(pos_scores >= 0) + torch.sum(neg_scores >= 0))
    recall = true_pos / len(pos_index)

    return mf_loss + emb_loss, mf_loss, emb_loss, precision, recall

def save_checkpoint(model: General, args: argparse.ArgumentParser, save_metric: float, epoch: int):
    keep_last_epochs = args.keep_last_epochs
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_ckpts = os.listdir(save_dir)
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
            all_ckpts.sort()
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
    with tqdm(total=n_test_batch) as t:
        t.set_description(f"Test Epoch {epoch_idx}")
        epoch_loss, epoch_mf_loss, epoch_emb_loss = 0, 0, 0
        epoch_total_precision, epoch_total_recall = 0, 0
        for batch_idx in range(1, n_test_batch + 1):
            author_path = paper_path = []
            author_embedding, paper_embedding = model(
                pretrained_author_embedding, 
                pretrained_paper_embedding,

            )
            test_pos_index, test_neg_index, test_authors, test_papers = data_generator.sample_test()
            test_loss, test_mf_loss, test_emb_loss, test_precision, test_recall = get_loss(author_embedding, paper_embedding, args.decay, test_pos_index, test_neg_index, test_authors, test_papers)
            
            epoch_loss += test_loss
            epoch_mf_loss += test_mf_loss
            epoch_emb_loss += test_emb_loss
            epoch_total_precision += test_precision
            epoch_total_recall += test_recall
    
    test_loss = epoch_loss / n_test_batch
    test_mf_loss = epoch_mf_loss / n_test_batch
    test_total_precision = epoch_total_precision / n_test_batch
    test_total_recall = epoch_total_recall / n_test_batch
    return test_loss, test_mf_loss, test_total_precision, test_total_recall
    pass

def train(model, optimizer, args):

    epoch = args.epoch
    begin_epoch = 1
    if os.path.exists(args.save_dir):
        ckpt_dir = os.listdir(args.save_dir)
        if ckpt_dir:
            last_model_dict = torch.load(os.path.join(args.save_dir, 'checkpoint_last.pt'))
            parameter_dict = last_model_dict['model_state']
            begin_epoch = last_model_dict['epoch'] + 1
            model.load_state_dict(parameter_dict)
    else:
        begin_epoch = 1
    for epoch_idx in range(begin_epoch, epoch + 1):
        n_train_batch = len(data_generator.real_train_index) // (data_generator.batch_size // 2) + 1
        with tqdm(total=n_train_batch) as t:
            t.set_description(f"Train Epoch {epoch_idx}")
            epoch_loss, epoch_mf_loss, epoch_emb_loss = 0, 0, 0
            epoch_total_precision, epoch_total_recall = 0, 0

            for batch_idx in range(1, n_train_batch + 1):
                # author_path, paper_path = data_generator.sample()
                author_path = paper_path = []
                author_embedding, paper_embedding = model(
                    pretrained_author_embedding, 
                    pretrained_paper_embedding,
                )
                # train_pos_index, train_neg_index, test_pos_index, test_neg_index, train_authors, train_papers, test_authors, test_papers = data_generator.get_train_test_indexes()
                train_pos_index, train_neg_index, train_authors, train_papers = data_generator.sample_train()
                loss, mf_loss, emb_loss, precision, recall = get_loss(author_embedding, paper_embedding, args.decay, train_pos_index, train_neg_index, train_authors, train_papers)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                epoch_mf_loss += mf_loss
                epoch_emb_loss += emb_loss
                epoch_total_precision += precision
                epoch_total_recall += recall
                t.set_postfix({"loss": f"{loss:.4f}", 'mf_loss': f"{mf_loss:.4f}", 'emb_loss': f"{emb_loss:.4f}", 'precision': f"{precision:.4f}", 'recall': f"{recall:.4f}"})
                t.update(1)
        print(f'Train Epoch {epoch_idx:.4f} Loss: {epoch_loss / n_train_batch:.4f} MF Loss: {epoch_mf_loss / n_train_batch:.4f} Emb Loss: {epoch_emb_loss / n_train_batch:.4f} Precision: {epoch_total_precision / n_train_batch:.4f} Recall: {epoch_total_recall / n_train_batch:.4f}')
        test_loss, test_mf_loss, test_total_precision, test_total_recall = test_one_epoch(model, args, epoch_idx)
        print(f'Test Epoch {epoch_idx:.4f} Loss: {test_loss:.4f} MF Loss: {test_mf_loss:.4f} Precision: {test_total_precision:.4f} Recall: {test_total_recall:.4f}')
        # n_test_batch = len(data_generator.real_test_index) // (data_generator.batch_size // 2) + 1
        # with tqdm(total=n_test_batch) as t:
        #     t.set_description(f"Test Epoch {epoch_idx}")
        #     epoch_loss, epoch_mf_loss, epoch_emb_loss = 0, 0, 0
        #     epoch_total_precision, epoch_total_recall = 0, 0
        #     for batch_idx in range(1, n_train_batch + 1):
        #         author_path, paper_path = data_generator.sample()
        #         author_embedding, paper_embedding = model(
        #             pretrained_author_embedding, 
        #             pretrained_paper_embedding,
        #             author_path,
        #             paper_path
        #         )
        #         test_pos_index, test_neg_index, test_authors, test_papers = data_generator.sample_test()
        #         test_loss, test_mf_loss, test_emb_loss, test_precision, test_recall = get_loss(author_embedding, paper_embedding, 0.1, test_pos_index, test_neg_index, test_authors, test_papers)
                
        #         epoch_loss += test_loss
        #         epoch_mf_loss += test_mf_loss
        #         epoch_emb_loss += test_emb_loss
        #         epoch_total_precision += test_precision
        #         epoch_total_recall += test_recall
        #         t.update(1)
        # print(f'Test Epoch {epoch_idx} Loss: {epoch_loss / n_test_batch} MF Loss: {epoch_mf_loss / n_test_batch} Emb Loss: {epoch_emb_loss / n_test_batch} Precision: {epoch_total_precision / n_test_batch} Recall: {epoch_total_recall / n_test_batch}')
        save_checkpoint(model, args, test_total_recall, epoch_idx)




if __name__ == '__main__':
    
    model = General(
        Pa_layers=args.pa_layers,
        Au_layers=args.au_layers,
        paper_adj=data_generator.paper_adj_matrix,
        author_adj=data_generator.author_adj_matrix,
        Paperdropout=args.gnn_dropout,
        Authordropout=args.gnn_dropout,
        n_authors=data_generator.n_authors,
        n_papers=data_generator.n_papers,
        num_layers=args.NGCF_layers,
        NGCFembed_dim=args.embed_dim,
        dropoutNGCF=args.ngcf_dropout,
        paper_dim=args.embed_dim,
        author_dim=args.embed_dim,
        norm_adj=data_generator.bipartite_lap_matrix,
        layer_size_list=args.layer_size_list,
        args=args
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, args)
