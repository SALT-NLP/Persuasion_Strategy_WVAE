import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from tqdm import tqdm

from sklearn.neighbors import KernelDensity

import gensim
import torch
import torch.utils.data as Data
import torchtext.vocab as vocab
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import *

from utils import sentence_tokenize, transform_format, check_ack_word

def read_data(data_path, n_labeled_data=300, n_unlabeled_data=-1, max_seq_num=8, max_seq_len=64, embedding_size=128, bert_encoder = False):
    
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
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    print('vocab size: ', tokenizer.vocab_size)

    np.random.seed(1)
    labeled_data_ids = list(labeled_data.keys())
    np.random.shuffle(labeled_data_ids)
    unlabeled_data_ids = list(unlabeled_data.keys())
    np.random.shuffle(unlabeled_data_ids)

    if len(labeled_data_ids) > 1000:
        n_labeled_data = min(len(labeled_data_ids)-800, n_labeled_data)
    else:
        n_labeled_data = min(len(labeled_data_ids)-500, n_labeled_data)
    
    train_labeled_ids = labeled_data_ids[:n_labeled_data]
    if n_unlabeled_data == -1:
        n_unlabeled_data = len(unlabeled_data_ids)
    train_unlabeled_ids = unlabeled_data_ids[:n_unlabeled_data]

    if len(labeled_data_ids) > 1000:
        val_ids = labeled_data_ids[-800:-400]
        test_ids = labeled_data_ids[-400:]
    else:
        val_ids = labeled_data_ids[-500:-300]
        test_ids = labeled_data_ids[-300:]
    
    train_labeled_dataset = Loader_labeled(
        tokenizer, labeled_data, train_labeled_ids, mid2target, label_mapping, max_seq_num, max_seq_len)
    train_unlabeled_dataset = Loader_unlabeled(
        tokenizer, unlabeled_data, train_unlabeled_ids, mid2target, max_seq_num, max_seq_len)

    val_dataset = Loader_labeled(
        tokenizer, labeled_data, val_ids, mid2target, label_mapping, max_seq_num, max_seq_len)
    test_dataset = Loader_labeled(
        tokenizer, labeled_data, test_ids, mid2target, label_mapping, max_seq_num, max_seq_len)

    n_class_sentence = 0
    for (u,v) in label_mapping.items():
        if v!= 0:
            n_class_sentence += 1
    n_class_sentence += 1

    doc_label = []
    for (u,v) in mid2target.items():
        doc_label.append(v)
    n_class_doc = max(doc_label) + 1

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}, N class {}, {}".format(
        len(train_labeled_ids), len(train_unlabeled_ids), len(val_ids), len(test_ids), n_class_sentence, n_class_doc))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, vocab_size, n_class_sentence, n_class_doc




    

class Loader_labeled(Dataset):
    def __init__(self, tokenizer, labeled_data, ids, target, label_set, max_seq_num=8, max_seq_len=64):
        self.tokenizer = tokenizer
        self.labeled_data = labeled_data
        self.ids = ids
        self.target = target
        self.max_seq_len = max_seq_len
        self.max_seq_num = max_seq_num

        self.label_set = label_set
        
        self.kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        
        self.load_data(labeled_data, ids)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        mid = self.ids[idx]
        sents, l, sent_len, doc_len= self.message[mid]

        message_target = self.lookup_score(mid)
        labels = np.array([10] * self.max_seq_num)
        doc_len = np.array(doc_len)
        sent_length = np.array([0] * self.max_seq_num)

        # select labeled sent
        mask1 = np.array([0] * self.max_seq_num)
        # select unlabeled sent
        mask2 = np.array([0] * self.max_seq_num)
        # select padded sent
        mask3 = np.array([1] * self.max_seq_num)
        # select unpadded sent
        mask4 = np.array([0] * self.max_seq_num)

        for i in range(0, len(l)):
            labels[i] = l[i]
            sent_length[i] = sent_len[i]
            if l[i] != 10:
                mask1[i] = 1
                mask2[i] = 0
                mask3[i] = 0
                mask4[i] = 1
            if l[i] == 10:
                mask1[i] = 0
                mask2[i] = 1
                mask3[i] = 0
                mask4[i] = 1
        
        message_vec = torch.LongTensor(self.message2id(sents))

        return (message_vec, labels, message_target, mask1, mask2, mask3, mask4, mid, sent_length, doc_len)

    
    def lookup_score(self, id):
        return self.target[id]
    
    def lookup_label_id(self, s):
        return self.label_set[s]
    
    def message2id(self, message):
        X = np.zeros([self.max_seq_num, self.max_seq_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_seq_num and j < self.max_seq_len:
                    id = self.tokenizer._convert_token_to_id(si)
                    X[i][j] = id
                    
        
        for i in range(len(message), self.max_seq_num):
            #print("....", self.tokenizer._convert_token_to_id('[PAD]'))
            
            X[i][0] = self.tokenizer._convert_token_to_id('[CLS]')
            X[i][1] = self.tokenizer._convert_token_to_id('[SEP]')

        return X
    
    def load_data(self, labeled_data, ids):
        self.message = {}
        
        labels_esit = []
        
        for i in tqdm(ids):
            sentences = []
            labels = []
            doc_len = []
            sent_len = []

            sents, l = labeled_data[i]

            for j in range(0, len(sents)):
                
                sents[j] = str(sents[j])
                
                results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", sents[j])
                results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", dd)
                results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                dd = results.sub(" <website> ", dd)

                #a = regexp_tokenize(transform_format(dd), self.pattern)
                tokens = self.tokenizer.tokenize(dd)

                temp = tokens
                if len(temp) > 0:
                    temp_ = ['[CLS]']
                    for k in range(0, min(len(temp), self.max_seq_len -2)):
                        temp_.append(temp[k])
                    temp_.append('[SEP]')
                    sentences.append(temp_)
                    
                    labels.append(self.lookup_label_id(l[j]))
                    
                    labels_esit.append(self.lookup_label_id(l[j]))
                    
                    sent_len.append(len(temp_) - 1)
            
            doc_len.append(len(sents) - 1)
            
            self.message[i] = (sentences, labels, sent_len, doc_len)  
            
        x_d = set()
        for (u, v) in self.label_set.items():
            x_d.add(v)
        x_d = np.array(list(x_d))
        
        
        self.kde.fit(np.array(labels_esit)[:, None])
        self.dist = self.kde.score_samples(x_d[:, None])

        
        self.esit_dist = F.softmax(torch.tensor(self.dist), dim = -1)
    


class Loader_unlabeled(Dataset):
    def __init__(self, tokenizer, unlabeled_data, ids, target, max_seq_num=8, max_seq_len=64):
        self.tokenizer = tokenizer
        self.unlabeled_data = unlabeled_data
        #self.ids = ids
        self.target = target
        self.max_seq_num = max_seq_num
        self.max_seq_len = max_seq_len

        self.load_data(unlabeled_data, ids)
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mid = self.ids[idx]
        sents, l, sent_len, doc_len  = self.message[mid]

        message_target = self.lookup_score(mid)
        
        doc_len = np.array(doc_len)

        sent_length = np.array([0] * self.max_seq_num)

        labels = np.array([10] * self.max_seq_num)
        # select labeled sent
        mask1 = np.array([0] * self.max_seq_num)
        # select unlabeled sent
        mask2 = np.array([0] * self.max_seq_num)
        # select padded sent
        mask3 = np.array([1] * self.max_seq_num)
        # select unpadded sent
        mask4 = np.array([0] * self.max_seq_num)
        for i in range(0, len(l)):
            labels[i] = l[i]
            sent_length[i] = sent_len[i]
            if l[i] != 10:
                mask1[i] = 1
                mask2[i] = 0
                mask3[i] = 0
                mask4[i] = 1
            if l[i] == 10:
                mask1[i] = 0
                mask2[i] = 1
                mask3[i] = 0
                mask4[i] = 1

        message_vec = torch.LongTensor(self.message2id(sents))

        return (message_vec, labels, message_target, mask1, mask2, mask3, mask4, mid, sent_length, doc_len)



    def message2id(self, message):
        X = np.zeros([self.max_seq_num, self.max_seq_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_seq_num and j < self.max_seq_len:
                    id = self.tokenizer._convert_token_to_id(si)
                    X[i][j] = id
                    
        
        for i in range(len(message), self.max_seq_num):
            X[i][0] = self.tokenizer._convert_token_to_id('[CLS]')
            X[i][1] = self.tokenizer._convert_token_to_id('[SEP]')

        return X
    
    def lookup_score(self, id):
        return self.target[id]
    
    def load_data(self, unlabeled_data, ids):
        self.message = {}
        self.ids = []
        self.data_num = 0

        for i in tqdm(ids):
            try:
                sentences = []
                labels = []
                doc = unlabeled_data[i]

                doc_len = []
                sent_len = []

                doc += '.'

                results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", doc)
                results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", dd)
                results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                dd = results.sub(" <website> ", dd)

                sents = sentence_tokenize(dd)

                # print(sents)

                for j in range(0, min(len(sents), self.max_seq_num)):
                    
                    tokens = self.tokenizer.tokenize(dd)

                    temp = tokens
                    if len(temp) > 0:
                        temp_ = ['[CLS]']
                        for k in range(0, min(len(temp), self.max_seq_len -2)):
                            temp_.append(temp[k])
                        temp_.append('[SEP]')
                        sentences.append(temp_)
                        labels.append(10)
                        sent_len.append(len(temp_) - 1)

                doc_len.append(min(len(sents) - 1, self.max_seq_num - 1))

                self.message[i] = (sentences[:self.max_seq_num],
                                   labels[:self.max_seq_num], sent_len[:self.max_seq_num], doc_len)
                self.ids.append(i)
                
            except:
                #print(doc)
                #exit()
                pass
    
