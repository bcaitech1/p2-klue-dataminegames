import pickle as pickle
import os
import pandas as pd
import torch

# convert to torch Dataset
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# preprocess datasets
def preprocessing_dataset(dataset, label_type):
    # modify = dict(pd.read_csv('/opt/ml/input/data/train/modify.txt', sep=' ', header=None).values)
    # dataset.loc[dataset[0].isin(modify.keys()), 8] = dataset[0].map(modify)
    if dataset[8][0] == 'blind':
        dataset[8] = 100
    else:
        dataset[8] = dataset[8].apply(lambda x: label_type[x])
        dataset = dataset[~dataset[8].isin((39,28,26,18,29,41,19,37,40))]
        
        
    dataset = dataset[[1,2,5,8]].rename(columns={1:'sentence', 2:'entity_01', 5:'entity_02', 8:'label'})
    return dataset

# load tsv datasets
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# tokenization: [CLS]entity1[SEP]entitiy2[SEP]sentence[SEP]
def tokenized_dataset(dataset, tokenizer):
    sep = tokenizer.special_tokens_map["sep_token"]
    concat_entity = list(dataset['entity_01'] + sep + dataset['entity_02'])
    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=180,
        add_special_tokens=True,
    )
    return tokenized_sentences
