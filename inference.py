from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device):
    dataloader = DataLoader(tokenized_sent, batch_size=1000, shuffle=False)
    model.eval()
    output_pred = []

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)

    return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_label = test_dataset['label'].values
    
    # tokenize dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return tokenized_test, test_label

def main(args):
    """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load tokenizer
    # bert-base-multilingual-cased
    # monologg/koelectra-base-v3-discriminator
    # xlm-roberta-large
    TOK_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

    # load model
    MODEL_NAME = args.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    pred_answer = inference(model, test_dataset, device)
    
    # make csv file with predicted answer
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./prediction/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default="./results/checkpoint-500")
    parser.add_argument('--seed', type=int, default=2021, help='seed (default = 2021)')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='model_name (default = xlm-roberta-large)')
    args = parser.parse_args()
    print(args)
    
    main(args)
  
