# -*- coding: utf-8 -*-
"""
TextClassification.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t-8yPIr3NwLpY0uRt4VEE74abSuOect1

# Text Classification Task

- Text Classification :
    - Embedding :
        - Random Word Embedding
        - Word2Vec Embedding
        - Glove Embedding
    - Model :
        - RNN
        - TextCNN
        - Bert

! gdown --id '1lz0Wtwxsh5YCPdqQ3E3l_nbfJT1N13V8' --output data.zip
! unzip data.zip
! ls
! pip install pytorch_pretrained_bert
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimz
from torch.utils.data import Dataset,DataLoader
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer,BertModel
import warnings
warnings.filterwarnings('ignore')

"""
Data Loader
"""

def load_train_data(path):
    if 'training_label' in path:
        with open(path, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' +++$+++ ') for line in lines]
        X = [line[1] for line in lines]
        Y = [int(line[0]) for line in lines]
        return X,Y
    else:
        with open(path, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
        return lines

def load_test_data(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        seq = [''.join(line.strip('\n').split(',')[1:]).strip() for line in lines[1:]]
    X = seq
    return X

def evaluation(pre, label):
    acc = np.sum(np.argmax(pre.cpu().data.numpy(), axis = 1) == label.numpy())
    return acc

"""
Tokenizer
"""

class Tokenizer():
    '''
    Data processing class,build the embedding vocab and turn the input sequence to the index in vocab.
    '''
    def __init__(self, config):
        self.embed_dim = config.embed_dim
        self.seq_len = config.max_len
        self.embed_type = config.embed_type
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word2vec = {}
        self.embedding_matrix = []
    
    # add <PAD> and <UNK> to embedding matrix
    def add_embedding(self, word):
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
        self.idx += 1

    # train the word2vec model
    def train_word2vec(self, data, window = 5):
        data = [line.strip('\n').split() for line in data]
        self.word2vec_model = word2vec.Word2Vec(data, size = self.embed_dim, window = window, iter = 10)

    # bulid the word2vec embedding matrix for RNN
    def build_word2vec_matrix(self):
        embedding_matrix = []
        for _,word in enumerate(self.word2vec_model.wv.vocab):
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            embedding_matrix.append(self.word2vec_model[word])
        self.embedding_matrix = torch.tensor(embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        assert self.embedding_matrix.shape[0] == self.idx
    
    # load the pre-trained Glove embedding vocab
    def load_embedding_vocab(self, path):
        with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
            for line in f:
                tokens = line.split()
                self.word2vec[tokens[0]] = np.asarray(tokens[1:], dtype = 'float32')

    # bulid the glove embedding matrix for RNN
    def build_glove_matrix(self,path):
        self.load_embedding_vocab(path)
        embedding_matrix = []
        for word in self.word2vec:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            embedding_matrix.append(self.word2vec[word])
            self.idx += 1
        self.embedding_matrix = torch.tensor(embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
        assert self.embedding_matrix.shape[0] == self.idx

    # build the embedding matrix
    def build_embed_matrix(self, data = None, path = None):
        if self.embed_type == 'w2v':
            self.train_word2vec(data)
            self.build_word2vec_matrix()
        elif self.embed_type == 'glove':
            self.build_glove_matrix(path)
        self.vocab_size = self.embedding_matrix.shape[0]

    # turn the input sequence to the index of vocab and pad the input sequence to the max length
    def seq_to_index(self, sequence):
        seq = [self.word2idx[word] if word in self.word2idx else self.word2idx['<UNK>'] for word in sequence]
        return seq

"""
DataSet
"""

class TextDataset():
    def __init__(self, config, tokenizer, X, y = None):
        self.X, self.Y = [],[]
        if config.model == 'bert':
            for item in X:
                seq = tokenizer.tokenize('[CLS] '+item + ' [SEP]')  
                seq = tokenizer.convert_tokens_to_ids(seq)
                text = self.padding_sequence(seq, config.max_len)
                self.X.append(text)          
        else:
            for item in X:
                seq = item.strip('\n').split()
                seq = tokenizer.seq_to_index(seq)
                seq = self.padding_sequence(seq, config.max_len)
                self.X.append(seq)   
        if y is not None:
            self.Y = torch.LongTensor(y)
            assert len(self.X) == len(self.Y)
        else:
            self.Y = y       
        
    def __getitem__(self, index):
        if self.Y is not None:
            return self.X[index],self.Y[index]
        else:
            return self.X[index]
    
    def __len__(self):
        return len(self.X)
    
    def padding_sequence(self, sentence, max_len):
        sentence = sentence[:max_len]
        out = (np.ones(max_len) * 0).astype('int64')
        out[:len(sentence)] = sentence
        return out

"""
RNN
"""

class RNN(nn.Module):

    def __init__(self, config, vocab_size, embed_matrix):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size,config.embed_dim)
        self.embed_dim = config.embed_dim
        if config.embed_matrix :
            self.embed.weight = nn.Parameter(torch.tensor(embed_matrix))
        if config.embed_grade :
            self.embed.weight.requires_grad = embed_grade
        
        self.LSTM = nn.LSTM(self.embed_dim, config.hid_dim, batch_first = True, dropout = config.dropout)
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hid_dim, 2 * config.hid_dim),
            nn.Linear(config.hid_dim * 2, config.out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        tokens = self.embed(torch.tensor(x))
        _,(ht,_) = self.LSTM(tokens)
        out = self.fc(ht)
        return out.reshape(batch_size,2)

"""
TextCNN
"""

class TextCNN(nn.Module):

    def __init__(self, config, vocab_size, embed_matrix, channel = 16):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, config.embed_dim)
        self.embed_dim = config.embed_dim
        self.channel = channel
        self.batch_size = config.batch_size
        if config.embed_matrix :
            self.embed.weight = nn.Parameter(torch.tensor(embed_matrix))
        if config.embed_grade :
            self.embed.weight.requires_grad = embed_grade

        self.cnn = nn.Sequential(  
            nn.Conv2d(1, channel, (3, self.embed_dim), stride = (1,1), padding = (1,0)),
            nn.MaxPool2d((config.max_len,1), stride = (1,1)),
            nn.Dropout(config.dropout),
        )
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.channel, config.out_dim)
        )
   
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(self.batch_size, self.channel).contiguous()
        out = self.dense(cnn_out)
        return out

"""
Bert
"""

class TCBert(nn.Module):

    def __init__(self, config, hidden_dim = 768):
        super(TCBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.out_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # bert_encode 可用于命名实体识别做序列标注，QA
        # bert_out 输出CLS信息，可用于文本分类
        bert_encode, bert_out = self.bert(inputs, output_all_encoded_layers = False)
        out = self.dense(bert_out)
        return out

"""
Configuration
"""

class Configuration():
    def __init__(self):
        # model initial
        self.model = 'bert'
        self.embed_dim = 100
        self.hid_dim = 512
        self.out_dim = 2
        self.embed_type = 'w2v'
        self.embed_matrix = False
        self.embed_grade = False
        # train process initial
        self.batch_size = 64
        self.max_len = 64
        self.lr = 1e-5
        self.epoch_num = 10
        self.dropout = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # data & model
        self.train_path = 'training_label.txt'
        self.unlabel_path = 'training_nolabel.txt'
        self.test_path = 'testing_data.txt'
        self.save_path = './'
        self.load_path = 'model_bert.pkl'
        self.load_model = True

    def update_load(self, path):
        self.load_model = True
        self.load_path = path

"""
Train
"""

def training(config, model, train, valid, optimizer, criterion):

    best_acc = 0
    train_len,val_len = len(train), len(valid)
    for epoch in range(config.epoch_num):
        # train
        model.train()
        train_acc, train_loss = 0,0
        for idx, (data, label) in enumerate(train):
            optimizer.zero_grad()
            train_pre = model(data.to(config.device))
            loss = criterion(train_pre, label.to(config.device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += evaluation(train_pre, label)

        # validation
        model.eval()
        with torch.no_grad():
            val_acc, val_loss = 0,0
            for idx, (data, label) in enumerate(valid):
                val_pre = model(data.to(config.device))
                loss = criterion(val_pre, label.to(config.device))
                val_loss += loss.item()
                val_acc += evaluation(val_pre, label)

        print('Epoch[%02d|%02d] | Train loss is : %.5f | Train acc is : %.3f | Valid loss is : %.5f | Valid acc is : %.3f '% \
            (epoch+1, config.epoch_num,train_loss / train_len , train_acc / (train_len*config.batch_size), val_loss / val_len, val_acc / (val_len*config.batch_size)))

        if val_acc/val_len > best_acc:
            best_acc = val_acc/val_len           
            print('***** The best accuracy in validation set is %.3f'%(val_acc / (val_len*config.batch_size)))
            save_model(config, model)

"""
Train Process
"""

def train_process(config, tokenizer):

    data, label = load_train_data(config.train_path)

    if config.model != 'bert':
        embed_matrix = tokenizer.embedding_matrix
        vocab_size = tokenizer.vocab_size
    
    train_data, val_data, train_label, val_label = train_test_split(data, label, train_size = 0.9)
    train = TextDataset(config, tokenizer, train_data, train_label)
    train_loader = DataLoader(train, batch_size = config.batch_size, shuffle = True)
    val = TextDataset(config, tokenizer, val_data, val_label)
    val_loader = DataLoader(val, batch_size = config.batch_size, shuffle = True)

  
    # DataLoader
    print('Data Loader build successfully.')

    if config.model == 'bert':
        model = TCBert(config)
    elif config.model == 'rnn':
        model = RNN(config, vocab_size, embed_matrix)
    else:
        model = TextCNN(config, vocab_size, embed_matrix)

    print(f'Model {config.model} initialize successfully.')

    model = model.to(config.device)
    optimizer = optimz.Adam(model.parameters(), lr = config.lr)
    criterion = nn.CrossEntropyLoss()
    
    training(config, model, train_loader, val_loader, optimizer, criterion)

"""
Save Model
"""

def save_model(config, model):
    path = os.path.join(config.save_path,f'model_{config.model}.pkl')
    torch.save(model, path)
    config.update_load(path)

"""
Load Model
"""

def load_model(config):
    if config.load_model:
        print(f'Load model path {config.load_path}')
        model = torch.load(config.load_path)
    return model

"""
Test
"""

def testing(model, test_loader, device, path):
    result = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            preds = model(data.to(device))
            result += np.argmax(preds.cpu().data.numpy(), axis = 1).tolist()
    outputs = pd.DataFrame({'id':[str(i) for i in range(len(test_loader)*config.batch_size)], 'label': result})
    outputs.to_csv(path, index = False)
    print('Save prediction csv successfully')

"""
Test Process
"""

def test_process(config, tokenizer):
    
    data = load_test_data(config.test_path)

    test_data = TextDataset(config, tokenizer, data)
    test_loader = DataLoader(test_data, batch_size = config.batch_size, shuffle = False)
    print('Load model ! ')
    model = load_model(config).to(config.device)

    path = f'{config.model}_submission.csv'

    testing(model, test_loader, config.device, path)

"""
 Build Tokenizer
 """

def build_tokenizer(config):

    if 'bert' == config.model:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = Tokenizer(config)
        train_data,_ = load_train_data(config.train_path)
        unlabel_data = load_train_data(config.unlabel_path)
        test_data = load_test_data(config.test_path)
        tokenizer.build_embed_matrix(data = test_data + unlabel_data + train_data)

    return tokenizer

"""
Main
"""

def main():
    config = Configuration()
    tokenizer = build_tokenizer(config)
    train_process(config, tokenizer)
    test_process(config, tokenizer)

if __name__ == '__main__':
    main()