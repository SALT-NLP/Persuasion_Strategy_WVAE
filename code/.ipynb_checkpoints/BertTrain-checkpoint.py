import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import Dataset
import math

from read_data_bert import *
from utils import *
from bertModel import *

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')

parser.add_argument('--data-path', type=str, default='./kiva/',
                    help='path to data folders')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0


def main():
    global best_acc
    count = 10
    train_labeled_set, val_set, test_set, n_labels = read_data(
        args.data_path, args.n_labeled)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=256, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=256, shuffle=False)

    model = ClassificationBert(n_labels).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])
    criterion = nn.CrossEntropyLoss()
    test_accs = []
    for epoch in range(args.epochs):
       
        train(labeled_trainloader, model, optimizer, criterion, epoch)


        val_acc, marcro = validate(
            val_loader, model, criterion, epoch,n_labels, mode='Valid Stats')
        print("epoch {}, macro {}, micro {}".format(
            epoch, val_acc, marcro))
        count = count - 1
        if val_acc >= best_acc:
            count = 10
            best_acc = val_acc
            test_acc, marcro = validate(
                test_loader, model, criterion, epoch, n_labels,mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, macro {},micro {}".format(
                epoch, test_acc, marcro))
        
        if count <= 0:
            print('early stop')
            break
        print('Best val_acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)
    

    print('Best val_acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)
    
    
    
def validate(valloader, model, criterion, epoch, n_labels, mode):
    model.eval()
    with torch.no_grad():
        predict_dict = {}
        correct_dict = {}
        correct_total = {}
        
        for i in range(0, n_labels):
            predict_dict[i] = 0
            correct_dict[i] = 0
            correct_total[i] = 0
        
        p = 0
        r = 0

        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, l = inputs.cuda(), targets.cuda(non_blocking=True)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, l)

            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(0, inputs.shape[0]):

                correct_total[np.array(l.cpu())[i]] += 1
                predict_dict[np.array(predicted.cpu())[i]] += 1

                if np.array(l.cpu())[i] == np.array(predicted.cpu())[i]:
                    correct_dict[np.array(l.cpu())[i]] += 1
                    

        f1 = []
        
        true_total_ = 0
        predict_total_ = 0
        correct_total_ = 0
        
        for (u, v) in correct_dict.items():
            if predict_dict[u] == 0:
                temp = 0
            else:
                temp = v/predict_dict[u]

            if correct_total[u] == 0:
                temp2 = 0
            else:
                temp2 = v/correct_total[u]
            
            if temp == 0 and temp2 == 0:
                f1.append(0)
            else:
                f1.append((2*temp*temp2)/(temp+temp2))
            
            true_total_ += correct_total[u]
            predict_total_ += predict_dict[u]
            correct_total_ += v
            
            #p += temp
            #r += temp2
        
        
        Marco_f1 = sum(f1)/(len(f1))
                            
        p =  correct_total_ / predict_total_
        r = correct_total_/ true_total_
              
        
        Micro_f1 = (2*p*r)/(p+r)
        
   
        print('true dist: ', correct_total)
        print('predict dist: ', predict_dict)
        print('correct pred: ', correct_dict)
        print('Macro: ', Marco_f1)
        print('Micro: ', Micro_f1)
 
    #return loss_total, acc_total, total_sample
    return Marco_f1,  Micro_f1

    
def train(labeled_trainloader, model, optimizer, criterion, epoch):
    labeled_train_iter = iter(labeled_trainloader)
    model.train()
    
    for batch_idx in range(args.val_iteration):
        try:
            inputs, targets  = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs, targets  = labeled_train_iter.next()

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    main()

