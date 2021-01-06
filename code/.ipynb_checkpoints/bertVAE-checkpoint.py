import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

class Classifier(nn.Module):
    def __init__(self, input_size, n_class):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

class Predictor(nn.Module):
    def __init__(self, n_class, out_class, z_size ,hard = True, hidden_size=64, num_layers=1, bidirectional=False):
        super(Predictor, self).__init__()
        self.z_size = z_size
        self.bidirectional = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hard = hard
        if hard:
            self.embedding = nn.Linear(n_class, z_size, bias=False)
            #self.embedding = nn.Embedding(n_class, z_size)
            
            #self.lstm = nn.LSTM(input_size=hidden_size + z_size, hidden_size=hidden_size,
            #                    num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            
            self.lstm = nn.LSTM(input_size=z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            
            #self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                #num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
                
                
            self.w_proj = nn.Linear(z_size, z_size)
            self.w_context_vector = nn.Parameter(torch.randn([z_size, 1]).float())
            self.softmax = nn.Softmax(dim = 2)
            
            
        else:
            self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.predict = nn.Linear(hidden_size, out_class)

    def forward(self, x, z, doc_len):
        # batch_size * seq_num * n_class
        batch_size = x.shape[0]
        
        seq_len = x.shape[1]
        
        # z_size 256
        # hidden_size 64
        #print(z.shape)

        if self.hard:
            x = self.embedding(x)
            # batch_size * seq_num * hidden_size
            
            #z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
            
            x = torch.cat([x.unsqueeze(2), z.unsqueeze(2)], dim=2)
            Hw = torch.tanh(self.w_proj(x))
            # batch_size *  seq_num * 2 * hidden_size
            w_score = self.softmax(Hw.matmul(self.w_context_vector))
            self.w_score = w_score
            x_out = x.mul(w_score)
            x_out = torch.sum(x_out, dim = 2)
            
            output, (h_n, c_n) = self.lstm(x_out)
            
            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), self.embedding.weight
        
        else:
            #z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
            x = torch.cat([x, z], dim=2)
            
            output, (h_n, c_n) = self.lstm(x)
            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), None


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')


    def forward(self, x, y=None, sent_len=None):
        # batch_size*seq_num * seq_len
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        all_hidden, pooler = self.bert(x)
        pooled_output = torch.mean(all_hidden, 1)

        return pooled_output 


class Generator(nn.Module):
    def __init__(self, vocab_size, z_size=128, embedding_size=768, generator_hidden_size=128, n_class=None, generator_layers=1):
        super(Generator, self).__init__()
        self.n_class = n_class
        self.generator_hidden_size = generator_hidden_size
        self.generator_layers = generator_layers
        self.vocab_size = vocab_size
        self.z_size = z_size

        if n_class is None:
            self.lstm = nn.LSTM(input_size=embedding_size + z_size,
                                hidden_size=generator_hidden_size, num_layers=generator_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=embedding_size + z_size + n_class,
                                hidden_size=generator_hidden_size, num_layers=generator_layers, batch_first=True)

        self.linear = nn.Linear(generator_hidden_size, vocab_size)

    def forward(self, x, z, y=None):
        # batch_size*seq_num * seq_len * embedding_size
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
        x = torch.cat([x, z], dim=2)

        if self.n_class is not None and y is not None:
            y = torch.cat([y] * seq_len, 1).view(batch_size,
                                                 seq_len, self.n_class)
            x = torch.cat([x, y], dim=2)

        output, hidden = self.lstm(x)
        
        output = self.linear(output)
        
        
        return output


class HierachyVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size=768, generator_hidden_size=1024, generator_layers=2, z_size=256, n_class=None, out_class = None, hard = True):
        super(HierachyVAE, self).__init__()
        
        self.z_size = z_size
        self.n_class = n_class

        self.encoder = Encoder()

        self.hidden_to_mu = nn.Linear(z_size + n_class, z_size)
        self.hidden_to_logvar = nn.Linear(z_size + n_class, z_size)
        
        self.hidden_linear = nn.Linear(768, z_size)
        
        self.classifier = nn.Linear(768, n_class)

        self.predictor = Predictor(n_class, out_class, z_size, hard)

        self.generator = Generator(
            vocab_size, z_size, embedding_size, generator_hidden_size, n_class, generator_layers)

    def encode(self, x, sent_len):
        encoder_hidden = self.encoder(x, sent_len = sent_len)
        
        #mu = self.hidden_to_mu(encoder_hidden)
        #logvar = self.hidden_to_logvar(encoder_hidden)
        
        q_y = self.classifier(encoder_hidden)
        
        encoder_hidden = self.hidden_linear(encoder_hidden)
        
        return q_y, encoder_hidden
        
    def sample_gumbel(self, logits, eps=1e-8):
        U = torch.rand(logits.shape)
        if logits.is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, tau):
        y = logits + self.sample_gumbel(logits)
        return F.softmax(y / tau, dim=-1)

    def gumbel_softmax(self, logits, tau, hard=True):
        y = self.gumbel_softmax_sample(logits, tau)

        if not hard:
            return y

        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, y.shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*y.shape)

        y_hard = (y_hard - y).detach() + y

        return y_hard.view(-1, y.shape[-1])

    def gaussian_sample(self, mu, logvar, batch_size):
        z = torch.randn([batch_size, self.z_size]).cuda()
        z = mu + z * torch.exp(0.5 * logvar)
        return z
    
    def forward(self, x, prob, tau, mask1, mask2, hard=True, y=None, doc_len = None, sent_len = None):
        batch_size = x.shape[0]
        seq_num = x.shape[1]
        seq_len = x.shape[2]
        
        n_labels = y.shape[-1]

        x = x.view(batch_size * seq_num, seq_len)
        y = y.view(batch_size*seq_num, self.n_class)
        mask1 = mask1.view(batch_size*seq_num)
        mask2 = mask2.view(batch_size * seq_num)

        sent_len = sent_len.view(batch_size * seq_num)

        q_y, encoder_hidden = self.encode(x, sent_len)
        
        

        y_sample = self.gumbel_softmax(q_y, tau, hard)

        y_in = y * mask1.view(-1, 1).float() + y_sample * mask2.view(-1, 1).float()
        
        
        #print(y_in.shape)
        #print(encoder_hidden.shape)
        
        hidden = torch.cat([encoder_hidden, y_in], dim = -1)
        
        mu = self.hidden_to_mu(hidden)
        logvar = self.hidden_to_logvar(hidden)
        
        y_in2 = y * mask1.view(-1,1).float() + F.softmax(q_y, dim=-1) * mask2.view(-1,1)
        
        y_in3 = F.softmax(q_y, dim=-1)

        z = self.gaussian_sample(mu, logvar, batch_size*seq_num)
        
        t, strategy_embedding = self.predictor(
            y_in2.view(batch_size, seq_num, self.n_class), encoder_hidden.view(batch_size, seq_num, self.z_size), doc_len)


        kld_z = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()

        input_shape = prob.size()
        device = prob.device 
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        embedding_output = self.encoder.bert.embeddings(
            input_ids=prob, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
        )

        logits = self.generator(embedding_output, z, y_in)
        
        
        return logits, kld_z, q_y, F.softmax(q_y, dim=-1), t, strategy_embedding









        
        