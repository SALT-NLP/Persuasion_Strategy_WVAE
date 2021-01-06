import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_size, n_class, hidden_size=64):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_class)
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
            self.embedding = nn.Linear(n_class, hidden_size, bias=False)
            #self.lstm = nn.LSTM(input_size=hidden_size + z_size, hidden_size=hidden_size,
                                #num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size=n_class + z_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        self.predict = nn.Linear(hidden_size, out_class)

    def forward(self, x, z, doc_len):
        # batch_size * seq_num * n_class
        batch_size = x.shape[0]
        
        seq_len = x.shape[1]
        
        #print(z.shape)

        if self.hard:
            #x = self.embedding(x)
            # batch_size * seq_num * hidden_size
            
            #z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
            x = torch.cat([x, z], dim=2)
            
            output, (h_n, c_n) = self.lstm(x)

            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), self.embedding.weight
        else:
            #z = torch.cat([z] * seq_len, 1).view(batch_size, seq_len, self.z_size)
            x = torch.cat([x, z], dim=2)
            
            output, (h_n, c_n) = self.lstm(x)
            hidden = output[torch.arange(output.shape[0]), doc_len.view(-1)]
            return self.predict(hidden), None


class Highway(nn.Module):
    def __init__(self, n_highway_layers, embedding_size):
        super(Highway, self).__init__()
        self.n_highway_layers = n_highway_layers

        self.non_linear = nn.ModuleList(
            [nn.Linear(embedding_size, embedding_size)] for _ in range(n_highway_layers))
        self.linear = nn.ModuleList(
            [nn.Linear(embedding_size, embedding_size)] for _ in range(n_highway_layers))
        self.gate = nn.ModuleList(
            [nn.Linear(embedding_size, embedding_size)] for _ in range(n_highway_layers))

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = F.sigmoid(self.gate[layer](x))
            non_linear = F.relu(self.non_linear[layer][x])
            linear = self.linear[layer](x)
            x = gate * non_linear + (1-gate) * linear

        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size=128, n_highway_layers=0, encoder_hidden_size=128, n_class=None, encoder_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_class = n_class

        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_layers = encoder_layers
        self.bidirectional = 2 if bidirectional else 1

        if n_class is None:
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=encoder_hidden_size,
                                num_layers=encoder_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_size=embedding_size + n_class, hidden_size=encoder_hidden_size,
                                num_layers=encoder_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, y=None, sent_len=None):
        # batch_size*seq_num * seq_len * embed_size
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        #x = self.highway(x)

        if self.n_class is not None and y is not None:
            y = torch.cat([y]*seq_len, 1).view(batch_size,
                                               seq_len, self.n_class)
            x = torch.cat([x, y], dim=2)

        output, (h_n, c_n) = self.lstm(x)

        #print('sent, ', output.shape)

        #hidden = output.transpose(1,0)[1]
        hidden = output[torch.arange(output.shape[0]), sent_len]
        #print(hidden.shape)
        
        #last_hidden = h_n
        #last_hidden = last_hidden.view(
        #    self.encoder_layers, self.bidirectional, batch_size, self.encoder_hidden_size)
        #last_hidden = last_hidden[-1]
        #hidden = torch.cat(list(last_hidden), dim=1)

        return hidden


class Generator(nn.Module):
    def __init__(self, vocab_size, z_size=128, embedding_size=128, generator_hidden_size=128, n_class=None, generator_layers=1):
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
    def __init__(self, vocab_size, embedding_size=128, n_highway_layers=0, encoder_hidden_size=128, encoder_layers=1, generator_hidden_size=128, generator_layers=1, z_size=128, n_class=None, out_class = None, bidirectional=False, pretrained_embedding=None, hard = True):
        super(HierachyVAE, self).__init__()
        self.z_size = z_size
        self.n_class = n_class

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if vocab_size is not None and pretrained_embedding is not None:
            pretrained_embedding = np.array(pretrained_embedding)
            self.embedding.weight.data.copy_(
                torch.Tensor(pretrained_embedding))

        #self.embedding.weight.requires_grad = False
        self.encoder = Encoder(embedding_size, n_highway_layers, encoder_hidden_size,
                                     n_class=None, encoder_layers=encoder_layers, bidirectional=bidirectional)

        double = 2 if bidirectional else 1
        
        #self.hidden_to_mu = nn.Linear(double * encoder_hidden_size, z_size)
        #self.hidden_to_logvar = nn.Linear(double * encoder_hidden_size, z_size)
        
        self.hidden_to_mu = nn.Linear(z_size + n_class, z_size)
        self.hidden_to_logvar = nn.Linear(z_size + n_class, z_size)
        
        self.hidden_linear = nn.Linear(double * encoder_hidden_size, z_size)
        
        self.classifier = nn.Linear(double * encoder_hidden_size, n_class)

        self.predictor = Predictor(n_class, out_class, z_size, hard)
        
        #self.predictor =nn.Linear(z_size, 1)

        self.generator = Generator(
            vocab_size, z_size, embedding_size, generator_hidden_size, n_class, generator_layers)

    def encode(self, x, sent_len):
        x = self.embedding(x)
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

        prob = self.embedding(prob)
        logits = self.generator(prob, z, y_in)
        
        return logits, kld_z, q_y, F.softmax(q_y, dim=-1), t, strategy_embedding









        
        