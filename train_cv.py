import pickle as pickle
from glob import glob
import os
import gc
import pandas as pd
import numpy as np
import random
import wandb
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import *
from importlib import import_module
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse


# metrics function for evaluation
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

# set fixed random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train(args):
    wandb.login()
    seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset_dir = "/opt/ml/input/data/train/train.tsv"
    dataset = load_data(train_dataset_dir)
    label  = dataset['label'].values  

    # k-fold cross validation
    cv = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)

    for idx , (train_idx , val_idx) in enumerate(cv.split(dataset, label)):
        # prepare tokenized datasets and labels each fold
        train_dataset = tokenized_dataset(dataset.iloc[train_idx], tokenizer)
        val_dataset = tokenized_dataset(dataset.iloc[val_idx], tokenizer)
        
        train_y = label[train_idx]
        val_y = label[val_idx]
        
        # make dataset for pytorch
        RE_train_dataset = RE_Dataset(train_dataset, train_y)
        RE_valid_dataset = RE_Dataset(val_dataset, val_y)
        
        # instantiate pretrained language model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=42)
        model.to(device)
        
        # callbacks
        early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.00005)
        
        # set training arguemnts
        output_dir = './result' + str(idx)
        training_args = TrainingArguments(
            output_dir=output_dir,          
            logging_dir='./logs',                         
            logging_steps=100,
            save_total_limit=1,
            evaluation_strategy='steps',
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            dataloader_num_workers=args.num_workers,
            fp16=True,

            seed=args.seed,
            run_name=args.run_name,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            label_smoothing_factor=args.label_smoothing_factor,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
        )
        
        # traniner
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=RE_train_dataset,         # training dataset
            eval_dataset= RE_valid_dataset,             # evaluation dataset
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,         # define metrics function
            callbacks=[early_stopping]
        )

        # train model
        trainer.train()
        
        # del model
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # del cache
        path = glob(f"/opt/ml/code/result{idx}/*")[0]
        for filename in os.listdir(path):
            if filename not in ['config.json', 'pytorch_model.bin', '.ipynb_checkpoints']:
                rm_filename = os.path.join(path, filename)
                os.remove(rm_filename)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2021, help='seed (default = 2021)')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='model_name (default = xlm-roberta-large)')
    parser.add_argument('--run_name', type=str, default='roberta', help='wandb run name (default = roberta)')
    parser.add_argument('--num_workers', type=int, default=4, help='CPU num_workers (default = 4)')
    parser.add_argument('--epochs', type=int, default=15, help='epochs (default = 15)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate(default = 2e-5)')
    parser.add_argument('--train_batch_size', type=int, default=32, help='train_batch_size (default = 32)')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='eval_batch_size (default = 32)')
    parser.add_argument('--warmup_steps', type=int, default=400, help='warmup_steps (default = 400)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay (default = 0.01)')
    parser.add_argument('--label_smoothing_factor', type=str, default=0.5, help='label_smoothing_factor (default = 0.5)')
    parser.add_argument('--early_stopping_patience', type=str, default=5, help='wandb run name (default = 5)')
    
    args = parser.parse_args()
    print(args)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    train(args)
