import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle

def read_data(data_path, n_labeled_data=300, max_seq_len=64):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
        # {mid: sentences, labels}
    with open(data_path + 'unlabeled_data.pkl', 'rb') as f:
        unlabeled_data = pickle.load(f)
        # {mid: message}
    with open(data_path + 'mid2target.pkl', 'rb') as f:
        mid2target = pickle.load(f)
        # {mid: target, team_size}
    
    with open(data_path + 'label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)

    print(label_mapping)
    '''
    try:
        with open(data_path + 'vocab_2.pkl', 'rb') as f:
            vocab = pickle.load(f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)
    except:
        vocab = Vocab(unlabeled_data=unlabeled_data,
                      labeled_data=labeled_data, embedding_size=embedding_size)

        with open(data_path + 'vocab_2.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)
    '''
    
    np.random.seed(1)
    labeled_data_ids = list(labeled_data.keys())
    np.random.shuffle(labeled_data_ids)
  

    if len(labeled_data_ids) > 1000:
        n_labeled_data = min(len(labeled_data_ids)-800, n_labeled_data)
    else:
        n_labeled_data = min(len(labeled_data_ids)-500, n_labeled_data)
    
    train_labeled_ids = labeled_data_ids[:n_labeled_data]
   
    if len(labeled_data_ids) > 1000:
        val_ids = labeled_data_ids[-800:-400]
        test_ids = labeled_data_ids[-400:]
    else:
        val_ids = labeled_data_ids[-500:-300]
        test_ids = labeled_data_ids[-300:]
    
    train_labeled_dataset = Loader_labeled(
        tokenizer, labeled_data, train_labeled_ids, mid2target, label_mapping, max_seq_len)
    val_dataset = Loader_labeled(
        tokenizer, labeled_data, val_ids, mid2target, label_mapping, max_seq_len)
    test_dataset = Loader_labeled(
        tokenizer, labeled_data, test_ids, mid2target, label_mapping, max_seq_len)

    n_class_sentence = 0
    for (u,v) in label_mapping.items():
        if v!= 0:
            n_class_sentence += 1
    n_class_sentence += 1

    doc_label = []
    for (u,v) in mid2target.items():
        doc_label.append(v)
    n_class_doc = max(doc_label) + 1

    print("#Labeled: {},  Val {}, Test {}, N class {}, {}".format(
        len(train_labeled_ids), len(val_ids), len(test_ids), n_class_sentence, n_class_doc))

    return train_labeled_dataset, val_dataset, test_dataset, n_class_sentence


class Loader_labeled(Dataset):
    def __init__(self, tokenizer, labeled_data, ids, target, label_set, max_seq_len=64):
        self.tokenizer = tokenizer
        self.labeled_data = labeled_data
        self.ids = ids
        self.max_seq_len = max_seq_len
        self.label_set = label_set
        self.load_data()
         
    def __len__(self):
        return len(self.labels)
    
    def lookup_label_id(self, s):
        return self.label_set[s]
        
    def __getitem__(self, idx):

        text = str(self.text[idx])
        
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return (torch.tensor(encode_result), self.labels[idx])
    
    def load_data(self):
        self.text = []
        self.labels = []
        for i in self.ids:
            sents, l = self.labeled_data[i]
            for j in range(0, len(sents)):
                self.text.append(sents[j])
                self.labels.append(self.lookup_label_id(l[j]))
               
            
                
      


