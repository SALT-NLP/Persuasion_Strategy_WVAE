import torch
import torch.nn as nn
from pytorch_transformers import *


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.max_pool = nn.MaxPool1d(256)
        
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels)
                                   )
        
       

    def forward(self, x, length = 256):

        all_hidden, pooler = self.bert(x)

        #if length == 256:
        #    pooled_output = self.max_pool(all_hidden.transpose(1, 2))
        pooled_output = torch.mean(all_hidden, 1)

        #pooled_output = pooled_output.squeeze(2)

        predict = self.linear(pooled_output)

        return predict