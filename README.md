## CS245 Final Project

### Author: S.Y Jiang, H.X Xu, and Y. Zhou

### Description
This repo is the final project for CS245: Foundations of Data Science in SJTU. We ranked 6th in the Kaggle leaderboard.
The structure of our model is as follows:
![None]()

### Data Preparation
Use the following command to generate preprocessed data. Assume `<datapath>` is where you store all preprocessed data and 
make sure that `data.zip` is unzipped to `<datapath>`.
```bash
python utils/prepare_data.py <datapath>
```

### Training
Use the following command the parameters to reproduce our model.
```bash
CUDA_VISIBLE_DEVICES=0 python train.py <checkpoints_dir> 
--batch_size 2048 \
--gat_layers 2 --lr 2e-4 --decay 1e-4 --epoch 100 
--datapath <datapath> \
--exp_name example --log_dir <tensorboard_dir>
```
```bash
CUDA_VISIBLE_DEVICES=0 python train.py <checkpoints_dir> 
--batch_size 2048 \
--gat_layers 2 --lr 2e-4 --decay 1e-5 --epoch 200 
--datapath <datapath> \
--exp_name example --log_dir <tensorboard_dir>
```
The saved 

### Evaluation
First run the following command to generate averaged checkpoint. `CKPT_DIR` is `<checkpoints_dir>` defined above:
```bash
python3 utils/average_checkpoints.py --inputs $CKPT_DIR --num-epoch-checkpoints 5 --output $CKPT_DIR/ckpt_avg5.pt
```
This script is referred from [fairseq](https://github.com/facebookresearch/fairseq).

Then use the following command to generate the score table in the checkpoints directory:
```bash
python evaluate.py $CKPT_DIR/ckpt_avg5.pt --output_dir $CKPT_DIR --output_name <output_name>
```
