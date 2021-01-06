# Persuasion_Strategy_WVAE

This repo contains codes for the following paper: 

*Jiaao Chen, Diyi Yang*: Weakly-Supervised Hierarchical Models for Predicting Persuasive Strategies in Good-faith Textual Requests. AAAI 2021

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will introduce the dataset and get you running the codes.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pytorch_transformers (also known as transformers)
* Pandas, Numpy, Scipy, Pickle


### Dataset

We annotated the sentences in datasets using the following strategy set:
```
{0: Others, 1: Credibility, 2: Reciprocity, 3: Evidence, 4: Commitment, 5: Scarcity, 6: Social Identity
, 7: Emotion, 8: Impact, 9: Politeness}
```
In each dataset, we merged rare strategies into Other category, resulting in "Evidence, Impact, Politeness, Reciprocity, Credibility" for Borrow, "Evidence, Impact, Politeness, Reciprocity, Scarcity,  Emotion" for RAOP, and "Evidence, Impact, Politeness, Credibility, Scarcity, Commitment" for Kiva.

The Borrow and RAOP datasets are in the `data` folder which follow the following structures:
```
|__ data/
        |__ borrow/ --> Datasets for Borrow
            |__ label_mapping.pkl --> dictionary that maps the annotated label id to label id used during training
            |__ labeled_data.pkl --> dictionary that maps the labeled message id to sentences in the message and their corresponding sentence-level annotated labels
            |__ unlabeled_data.pkl --> dictionary that maps the unlabeled message id to sentences in the message
            |__ mid2target.pkl --> dictionary that maps message id to the document level labels
            |__ vocab.pkl --> vocabulary files for Borrow dataset

        |__ raop/ --> Datasets for Random Acts of Pizza
            |__ label_mapping.pkl --> dictionary that maps the annotated label id to label id used during training
            |__ labeled_data.pkl --> dictionary that maps the labeled message id to sentences in the message and their corresponding sentence-level annotated labels
            |__ unlabeled_data.pkl --> dictionary that maps the unlabeled message id to sentences in the message
            |__ mid2target.pkl --> dictionary that maps message id to the document level labels
            |__ vocab.pkl --> vocabulary files for RAOP dataset
```



### Training models
This section contains instructions for training WS-VAE-BERT models on Borrow and RAOP using 20 labeled message for training.

#### Training WS-VAE-BERT on Borrow
```
python ./code/train_bert_vae.py --epochs 50 --batch-size 4 --batch-size-u 6 --val-iteration 240 \
--gpu 0 --max-seq-len 64 --max-seq-num 7 --data-path ./data/borrow/ \
--rec-coef 1 --predict-weight 0.1 --class-weight 5 --kld-weight-y 1 --kld-weight-z 1 \
--word-dropout 0.6 --kld-y-thres 1.2 --warm-up False --tsa-type no --hard True \
--n-labeled-data 20 
```


#### Training WS-VAE-BERT on RAOP
```
python ./code/train_bert_vae.py --epochs 50 --batch-size 4 --batch-size-u 6 --val-iteration 240 \
--gpu 0 --max-seq-len 64 --max-seq-num 8 --data-path ./data/raop/ \
--rec-coef 1 --predict-weight 0.1 --class-weight 5 --kld-weight-y 1 --kld-weight-z 1 \
--word-dropout 0.6 --kld-y-thres 1.2 --warm-up False --tsa-type no --hard True \
--n-labeled-data 20
```







