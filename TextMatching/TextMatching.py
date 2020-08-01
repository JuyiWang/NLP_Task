# -*- coding: utf-8 -*-
"""
TextMatching.py

- Text Matching Task
    - Embedding :
        - Randomly Initialize Embedding
        - Word2vec Embedding
        - Glove Embedding
    - Model :
        - ESIM
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimz
from torch.utils.data import Dataset,DataLoader

"""
Data Loader
"""

def data_loader(path):
    X = []
    Y = []
    dic = pd.read_csv(path, sep = '\t')
    idx = 0
    for index, row in dic.iterrows():
        # print(index, row['gold_label'], row['sentence1'], row['sentence2'])
        if row['gold_label'] != 'contradiction' and row['gold_label'] != 'neutral' and row['gold_label'] != 'entailment':
            idx += 1
        else:
            if row['gold_label'] == 'entailment':
                X.append((row['sentence1'],row['sentence2']))
                Y.append(1)
            elif row['gold_label'] == 'contradiction':
                X.append((row['sentence1'],row['sentence2']))
                Y.append(2)
            else:
                X.append((row['sentence1'],row['sentence2']))
                Y.append(0)
    assert len(X) == len(Y),'Data load fail.'
    print(f'Valid data {len(X)}, invalid data {idx}')
    return X,Y

"""
Tokenizer
"""

class Tokenizer():
    def __init__(self, config):
        self.embed_dim = config.embed_dim
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
        if word == '<UNK>':
            torch.nn.init.xavier_normal_(vector)
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
        assert self.embedding_matrix.shape[0] == self.idx,'Word vector indexes do not match'
    
    # load the pre-trained Glove embedding vocab
    def load_embedding_vocab(self, path):
        with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
            for line in f:
                tokens = line.split()
                self.word2vec[' '.join(tokens[:-300])] = np.asarray(tokens[-300:], dtype = 'float32')

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
        assert self.embedding_matrix.shape[0] == self.idx,'Word vector indexes do not match'

    # build the embedding matrix
    def build_embed_matrix(self, data = None, path = None):
        if self.embed_type == 'w2v':
            self.train_word2vec(data)
            self.build_word2vec_matrix()
        elif self.embed_type == 'glove':
            self.build_glove_matrix(path)
        print('embed matrix load successfully')
        self.vocab_size = self.embedding_matrix.shape[0]

    # turn the input sequence to the index of vocab and pad the input sequence to the max length
    def convert_tokens_to_ids(self, sequence):
        seq = [self.word2idx[word] if word in self.word2idx else self.word2idx['<UNK>'] for word in sequence]
        return seq

"""
DataSet
"""

def MatchDataset():
    def __init__(self, config, tokenizer, X, y = None):
        self.X,self.Y = [],[]
        for item in X:
            sentence1 = item[0].strip('\n').split()
            sentence2 = item[1].strip('\n').split()
            sentence1 = tokenizer.convert_tokens_to_ids(sentence1)
            sentence2 = tokenizer.convert_tokens_to_ids(sentence2)
            sentence1 = self.padding_sequence(sentence1, config.max_len)
            sentence2 = self.padding_sequence(sentence2, config.max_len)
            self.X.append((sentence1, sentence2))
        if y is not None:
            self.Y = torch.LongTensor(y)
            assert len(self.X) == len(self.Y),'Data and label do not match'
        else:
            self.Y = y

    def __getitem__(self, index):
        return self.X[index],self.Y[index] if self.Y is not None else self.X[index]

    def __len__(self):
        return len(self.X)

    def padding_sequence(self, sentence, max_len):
        sentence = sentence[:max_len]
        out = (np.ones(max_len) * 0).astype('int64')
        out[:len(sentence)] = sentence
        return out

"""
Model
"""

class ESIM(nn.Module):
    def __init__(self, config, vocab_size, embed_matrix):
        super(ESIM, self).__init__()
        # Input encoding
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.enc_lstm = nn.LSTM(config.embed_dim, config.hid_dim, batch_first = config.dropout, bidirectional = True)
        # Local inference modeling
        # self.attention()
        # Inference composition
        self.dense = nn.Sequential(
            nn.Linear(config.hid_dim * 4, config.hid_dim),
            nn.ReLU()
        )
        self.infer_lstm = nn.LSTM(config.hid_dim, config.hid_dim, batch_first = config.dropout, bidirectional = True)
        # Prediction
        self.ave_pool = nn.AvgPool2d((3,config.hid_dim*2),(1,0),padding = (1,0))
        self.max_pool = nn.MaxPool2d((3,config.hid_dim*2),(1,0),padding = (1,0))
        self.MLP = nn.Sequential(
            nn.linear(config.hid_dim * 2, config.hid_dim * 4),
            nn.Tanh(),
            nn.Linear(config.hid_dim * 4, config.out_dim),
            nn.Softmax()
        )
    
    def forward(self, inputs):
        # Input encoding
        pre,hyp = inputs[0],inputs[1]
        pre_emb = self.embedding(pre)
        hyp_emb = self.embedding(hyp)
        pre_enc,_ = self.enc_lstm(pre_emb)
        hyp_enc,_ = self.enc_lstm(hyp_emb)
        '''
        pre_enc | hyp_enc = [batch, seq_len, num_directions * hidden_size]
        '''
        # Local Inference
        pre_attn = self.attention(pre_enc, hyp_enc, hyp_enc)
        hyp_attn = self.attention(hyp_enc, pre_enc, pre_enc)
        m_pre = torch.cat((pre_enc, pre_attn,pre_enc - pre_attn, pre_enc * pre_attn), dim = -1)
        m_hyp = torch.cat((hyp_enc, hyp_attn,hyp_enc - pre_attn, hyp_enc * pre_attn), dim = -1) 
        #Inference Composition
        infer_pre = self.infer_lstm(self.dense(m_pre))
        infer_hyp = self.infer_lstm(self.dense(m_pre))
        ave_pre,max_pre = self.ave_pool(infer_pre),self.max_pool(infer_pre)
        ave_hyp,max_hyp = self.ave_pool(infer_hyp),self.max_pool(infer_hyp)
        vec = torch.cat((ave_pre,max_pre,ave_hyp,max_hyp))
        # Prediction
        out = self.MLP(vec)
        return out

    def attention(self,q,k,v):
        k = k.transpose(0,2,1).contiguous()
        p_a = torch.matmul(p,k)
        p_a = nn.Softmax(p_a, dim = 2)
        vec = torch.matmul(p_a, v)
        return vec

"""
Training
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
Train process
"""

def train_process(config, tokenizer):  
    train_data, train_label = data_loader(config.train_path)
    val_data = data_loader(config.valid_path)

    embed_matrix = tokenizer.embedding_matrix
    vocab_size = tokenizer.vocab_size

    train = MatchDataset(config, tokenizer, train_data, train_label)
    train_loader = DataLoader(train, batch_size = config.batch_size, shuffle = True)
    val = MatchDataset(config, tokenizer, val_data, val_label)
    val_loader = DataLoader(val, batch_size = config.batch_size, shuffle = True)
    # DataLoader
    print('Data Loader build successfully.')
    model = ESIM(config, vocab_size, embed_matrix)
    model = model.to(config.device)
    print(f'Model {config.model} initialize successfully.')

    optimizer = optimz.Adam(model.parameters(), lr = config.lr)
    criterion = nn.CrossEntropyLoss()
    training(config, model, train_loader, val_loader, optimizer, criterion)   

"""
Save model
"""

def save_model(config, model):
    path = os.path.join(config.save_path,f'model_{config.model}.pkl')
    torch.save(model, path)
    config.update_load(path)

"""
Load model
"""

def load_model(config):
    if config.load_model:
        print(f'Load model path {config.load_path}')
        model = torch.load(config.load_path)
    return model   

"""
Configuration
"""

class Configuration():
    def __init__(self):
        # model initial
        self.model = 'ESIM'
        self.embed_dim = 300
        self.hid_dim = 512
        self.out_dim = 3
        self.embed_type = 'w2v'
        self.embed_matrix = False
        self.embed_grade = False
        # train process initial
        self.batch_size = 32
        self.max_len = 64
        self.lr = 4e-4
        self.epoch_num = 10
        self.dropout = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # data & model
        self.train_path = 'snli_1.0_train.txt'
        self.valid_path = 'snli_1.0_dev.txt'
        self.test_path = 'snli_1.0_test.txt'
        self.save_path = './'
        self.load_path = 'model_bert.pkl'
        self.load_model = True

    def update_load(self, path):
        self.load_model = True
        self.load_path = path

"""
Bulid Tokenizer
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

def main(config):
    config = Configuration()
    tokenizer = build_tokenizer(config)
    train_process(config, tokenizer)

if __name__ == '__main__':
    main()
