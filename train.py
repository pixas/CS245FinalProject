import argparse
import os
from models.General import General
import time
import torch
import torch.nn.functional as F 
import torch.optim as optim
from tqdm import tqdm
from utils.data import Data


TRAIN_FILE_TXT = 'data/bipartite_train.txt'

# TODO: load from file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = Data(random_walk_length=16,device=device)
pretrained_author_embedding = data_generator.author_embeddings
pretrained_paper_embedding = data_generator.paper_embeddings


def get_loss(author_embedding, paper_embedding, decay):
    author_embedding = F.normalize(author_embedding, p=2, dim=1)
    paper_embedding = F.normalize(paper_embedding, p=2, dim=1)
    score_matrix = torch.matmul(author_embedding, paper_embedding.transpose(0, 1))
    train_scores = score_matrix[list(zip(*data_generator.train_index))]
    mf_loss = torch.sum(1 - train_scores) / (len(train_scores))

    train_users = author_embedding[data_generator.train_authors]
    train_papers = paper_embedding[data_generator.train_papers]
    regularizer = (torch.norm(train_users) ** 2 + torch.norm(train_papers) ** 2) / 2
    emb_loss = decay * regularizer / (data_generator.train_author_cnt + data_generator.train_paper_cnt)
    
    # pred_pos = torch.sum(score_matrix >= 0.5)
    true_pos = torch.sum(train_scores >= 0)
    precision = 0
    recall = true_pos / len(data_generator.train_index)

    return mf_loss + emb_loss, mf_loss, emb_loss, precision, recall

def save_checkpoint(model: General, save_dir: str, keep_last_epochs: int, save_metric: float, epoch: int):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_ckpts = os.listdir(save_dir)
    cur_save_info = {
            'model_state': model.state_dict(),
            'metric': save_metric,
            'epoch': epoch
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


def train(model, optimizer, args):
    epoch = args.epoch
    with tqdm(total=epoch) as t:
        t.set_description("Training")
        for epoch_idx in range(1, epoch + 1):
            t1 = time.time()
            author_path, paper_path = data_generator.sample()
            author_embedding, paper_embedding = model(
                pretrained_author_embedding, 
                pretrained_paper_embedding,
                author_path,
                paper_path
            )
            loss, mf_loss, emb_loss, precision, recall = get_loss(author_embedding, paper_embedding, 0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2 = time.time()
            t.set_postfix({'epoch': f"{epoch_idx:>02d}", "loss": f"{loss:.4f}", 'mf_loss': f"{mf_loss:.4f}", 'emb_loss': f"{emb_loss:.4f}", "time": f"{t2-t1:.4f}s", 'precision': f"{precision:.4f}", 'recall': f"{recall:.4f}"})
            t.update(1)
            save_checkpoint(model, args.save_dir, args.keep_last_epochs, recall, epoch_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Run General")
    parser.add_argument('save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--module_type', nargs='?', default='LSTM', help='Module in coauthor and citation network')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--keep_last_epochs', type=int, default=5, help='only keep last epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model = General(
        RWembed_dim=512,
        stack_layers=2,
        dropoutRW=0.3,
        n_authors=data_generator.n_authors,
        n_papers=data_generator.n_papers,
        num_layers=2,
        NGCFembed_dim=512,
        dropoutNGCF=0.3,
        paper_dim=512,
        author_dim=512,
        norm_adj=data_generator.bipartite_lap_matrix,
        n_fold=4,
        args=args
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, args)