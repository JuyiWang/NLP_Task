# -*- coding: utf-8 -*-
"""

MachineTranslation.py

- Machine Trasnlation Task :
    - Embedding :
        - Word2vec Embedding
    - Model :
        - Seq2seq with attention
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimz
import numpy as np
import pandas as pd
import jieba
import random
from torch.utils.data import Dataset,DataLoader
from gensim.models import word2vec
import re
import warnings
warnings.filterwarnings('ignore')

"""
Data preprocess
"""

class data_preprocess():
    def __init__(self, config):
        self.root = config.data_path
        self.embed_dim = config.embed_dim
        self.cn_int2word, self.cn_word2int = self.get_dictionary('cn')
        self.en_int2word, self.en_word2int = self.get_dictionary('en')

    def get_dictionary(self, language):
        """
        Load the vocabulary
        return:
            int2word, word2int : dic of vocabulary
        """
        with open(os.path.join(self.root, f'int2word_{language}.json'), 'r') as f:
            int2word = json.load(f)
        with open(os.path.join(self.root, f'word2int_{language}.json'), 'r') as f:
            word2int = json.load(f)
        return int2word, word2int

    def train_word2vec(self, en_data, cn_data):
        """
        Train the word2vec model
        return:
            None
        """
        self.en_word2vec = word2vec.Word2Vec(en_data, size = self.embed_dim, window = 5, min_count = 2)
        self.cn_word2vec = word2vec.Word2Vec(cn_data, size = self.embed_dim, window = 5, min_count = 2) 

    def added_embedding(self):
        """
        Initial tag embedding <PAD><BOS><EOS><UNK>
        """
        pad_vector = torch.empty(1, self.embed_dim)
        vector = torch.empty(3, self.embed_dim)
        torch.nn.init.uniform_(vector)
        return torch.cat([pad_vector, vector], 0)

    def build_embedding(self):
        """
        Build the embedding matrix for nn.Embedding
        return:
            en_embedding,cn_embedding : tensor of vocab_size * embed_dim
        """
        en_add_embedding = self.added_embedding()
        en_embedding = np.empty((len(self.en_word2int), self.embed_dim))
        for word in self.en_word2int:
            en_embedding[self.en_word2int[word],:] = self.en_word2vec.wv[word] if word in self.en_word2vec.wv.vocab else en_add_embedding[3,:]
        self.en_embed_matrix = torch.cat([en_add_embedding, torch.tensor(en_embedding[4:],dtype = torch.float)], 0) 

        cn_add_embedding = self.added_embedding()
        cn_embedding = np.empty((len(self.cn_word2int), self.embed_dim))
        for word in self.cn_word2int:
            cn_embedding[self.cn_word2int[word],:] = self.cn_word2vec.wv[word] if word in self.cn_word2vec.wv.vocab else cn_add_embedding[3,:]
        self.cn_embed_matrix = torch.cat([cn_add_embedding, torch.tensor(cn_embedding[4:],dtype = torch.float)], 0)
        return self.en_embed_matrix, self.cn_embed_matrix
    
    def pad_sequence(self, seq, max_len):
        """
        Padding sequence '[BOS] sequences [EOS]' fixed length = max_len
        return:
            fixed length sequence
        """
        seq.insert(0,1)
        seq_out = [0]*max_len
        seq = seq[:max_len-1]
        seq.append(2)
        seq_out[:len(seq)] = seq
        return seq_out

    def seq_preprocess(self, sequence, max_len):
        """
        Take the input word sequnence to index sequence and padding to fixed length
        return:
            en_out,cn_out : list of fixed length
        """
        x = sequence.strip('\n').split('\t')
        # word to index 
        en_idx = [self.en_word2int[word] if word in self.en_word2int else self.en_word2int['<UNK>'] for word in x[0].split()]
        cn_idx = [self.cn_word2int[word] if word in self.cn_word2int else self.cn_word2int['<UNK>'] for word in x[1].split()]
        
        en_out = self.pad_sequence(en_idx, max_len)
        cn_out = self.pad_sequence(cn_idx, max_len)  
        return en_out,cn_out

class LabelTransform(object):
  def __init__(self, size, pad):
    self.size = size
    self.pad = pad

  def __call__(self, label):
    label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
    return label

"""
Load word2vec data
"""

def load_train_data(path):
    """
    Load the corpis to train the word2vec
    return:
        en_data, cn_data : corpus for word2vec training
    """
    with open(path, 'r') as f:
        en_data,cn_data = [],[]
        lines = f.readlines()
        for line in lines:
            seq = line.strip('\n').split('\t')
            en_data.append(seq[0].split())
            cn_data.append(seq[1].split()) 
        return en_data,cn_data

"""
Build data process
"""

def build_data_process(path_list, config):
    """
    return:
        process : class data_preprocess 
        en_embedding : english word2vec | array of vocab_size * embed_dim
        cn_embedding : chinese word2vec | array of vacab_size * embed_dim
    """
    en_train_vec, cn_train_vec = [],[]
    for path in path_list:
        en_data, cn_data = load_train_data(path)
        en_train_vec += en_data
        cn_train_vec += cn_data
    
    process = data_preprocess(config)
    process.train_word2vec(en_train_vec, cn_train_vec)
    en_embedding,cn_embedding = process.build_embedding()
    return process, en_embedding, cn_embedding

class TextDataSet(Dataset):
    def __init__(self, path, process, max_len):
        with open(path, 'r') as f:
            self.en, self.cn = [],[]
            for line in f.readlines():
                en_seq, cn_seq = process.seq_preprocess(line, max_len)
                self.en.append(en_seq)
                self.cn.append(cn_seq)
            assert len(self.cn) == len(self.en)
    
    def __len__(self):
        return len(self.en)

    def __getitem__(self,index):
        return np.array(self.en[index]), np.array(self.cn[index])

"""
Model
Encoder
"""

class Encoder(nn.Module):
    # Encoder Model
    def __init__(self, embed_matrix, layers = 1, hidden_dim = 128):
        super(Encoder, self).__init__()
        self.embed_dim = embed_matrix.shape[1]
        self.embed = nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1])
        self.rnn = nn.GRU(self.embed_dim, hidden_dim ,num_layers = layers, batch_first = True,bidirectional = True)

    def forward(self, inputs):
        inputs_len = torch.tensor(torch.sum(inputs != 0,dim = -1),dtype = torch.float)
        tokens = self.embed(inputs)
        out,h_t = self.rnn(tokens)
        return out,h_t

"""
Attention
"""

class Attention(nn.Module):
    # Attention Model
    def __init__(self, enc_dim, dec_dim, num_layers):
        super(Attention,self).__init__()
        self.enc_dim = enc_dim * 2
        self.dec_dim = dec_dim * 2
        self.softmax = nn.Softmax(dim = 2)
        self.dense1 = nn.Linear(self.enc_dim+self.dec_dim,self.dec_dim, bias = False)
        self.W_q = nn.Linear(self.enc_dim * num_layers, self.enc_dim)

    def forward(self, k, q, v):
        q = self.W_q(q)
        q_t = q.permute(1,2,0).contiguous()
        score = torch.matmul(k, q_t)
        score = score.permute(0,2,1).contiguous()
        out = torch.matmul(self.softmax(score), v)
        q = q.permute(1,0,2).contiguous()   
        out = self.dense1(torch.cat([q, out], dim = -1))
        return out

"""
Decoder
"""

class Decoder(nn.Module):
    # Decoder Model
    def __init__(self, embed_matrix, attention, layers = 1, hidden_dim = 128, dropout = 0.5):
        super(Decoder, self).__init__()
        self.embed_dim = embed_matrix.shape[1]
        self.out_dim = embed_matrix.shape[0]
        self.hid_dim = hidden_dim * 2
        # self.embedding = nn.Embedding.from_pretrained(embed_matrix)
        self.embedding = nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1])
        self.attn = attention
        self.RNN = nn.GRU(self.embed_dim, self.hid_dim, num_layers = layers,batch_first = True, bidirectional = False)
        self.dense = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(self.hid_dim, self.hid_dim*2),
                        nn.Linear(self.hid_dim*2, self.hid_dim*4),
                        nn.Linear(self.hid_dim*4, self.out_dim),
                    )

    def forward(self, inputs, hidden, encoder_out):
        inputs = inputs.unsqueeze(1)
        tokens = self.embedding(inputs)
        # rnn_out : [batch, seq_len, num_directions * hidden_size]
        # h_t : [num_layers * num_directions, batch, hidden_size]
        rnn_out, h_t = self.RNN(tokens, hidden)
        h_in = h_t.view(1,tokens.shape[0],-1)
        attn_out = self.attn(encoder_out, h_in, encoder_out)
        attn_out = attn_out.expand(-1,rnn_out.shape[1],-1)
        out = self.dense(attn_out.squeeze(1)) 
        return out, h_t

"""
Scheduler sampling
"""

def scheudle_sampling(Type,epoch = 1):
    if Type == 'train':
        return 1 - 0.03 * epoch
    else:
        return -1

"""
Seq2Seq
"""

class Seq2Seq(nn.Module):
    # Sequence to Sequence Model
    def __init__(self, encoder, decoder, device, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_layers = num_layers

    def forward(self, inputs, target, teacher_forcing_ratio):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.out_dim

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        preds = []
        # encoder_out : [batch, seq_len, num_directions * hidden_size]
        # encoder_ht  : [num_layers * num_directions, batch, hidden_size]
        encoder_out, hidden = self.encoder(inputs)
        hidden = hidden.view(self.num_layers, 2, batch_size, -1).contiguous()
        hidden = torch.cat((hidden[:,-2,:,:], hidden[:,-1,:,:]), dim = 2)

        decoder_in = target[:,0]
        preds = []
        for t in range(1,target_len):
            output, hidden = self.decoder(decoder_in, hidden, encoder_out)           
            outputs[:,t] = output
            top = output.argmax(1)
            teacher_force = random.random() <= teacher_forcing_ratio
            # 如果是 teacher force 则用target训练，否则使用prediction做预测
            decoder_in = target[:,t] if teacher_force and t < target_len else top
            preds.append(top.unsqueeze(1))
        preds = torch.cat(preds, 1)

        return outputs,preds

"""
Utils
- Basic operation
    - Save Model
    - Load Model
    - Build Model
    - Tokens to Sequence
    - Compute BLEU score
"""

def save_model(model, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.cpkt')

"""
Load Model
"""

def load_model(model, load_model_path):
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    return model

"""
Build Model
"""

def build_model(config,en_embed,cn_embed):
    attention = Attention(config.hidden_dim, config.hidden_dim, config.num_layers)
    encoder = Encoder(en_embed, layers = config.num_layers, hidden_dim = config.hidden_dim)
    decoder = Decoder(cn_embed, attention, layers = config.num_layers, hidden_dim = config.hidden_dim, dropout = config.dropout)
    model = Seq2Seq(encoder,decoder,config.device, config.num_layers)
    optimizer = optimz.Adam(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    
    if config.load_model:
        model = load_model(model, config.load_model_path)

    model = model.to(config.device)
    return model,optimizer

"""
Tokens to Sequence
"""

def tokens_to_sequence(outputs, int2word):
    """
    Transform tokens into a sequence
    return :
        sentence of word character
    """
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    return sentences

"""
Compute BLEU score
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def compute_bleu(sentences, targets):
    """
    Compute the BLEU score between the predict sequence and target
    return :
        BLEU score of one batch
    """
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        """
        Split the sentence into character level list
        return:
            tmp : the list of character tokens
        """
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding = 'utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp
    
    for sentence, target in zip(sentences, targets):
        sentenc = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(0.25,0.25,0.25,0.25))
    return score

"""
Config
"""

class Config(object):
    def __init__(self):
        self.batch_size = 64
        self.embed_dim = 256
        self.hidden_dim = 512
        self.num_layers = 3
        self.dropout = 0.5
        self.learning_rate = 5e-4
        self.weight_decay = 0
        self.epoch_num = 40
        self.max_len = 16
        self.load_model = False
        self.load_model_path = ''
        self.store_model_path = "./ckpt"      
        self.data_path = "./cmn-eng"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
rain & Test
Train model
"""

def train(model, optimizer, train_iter, loss_func, device, epoch):
    model.train()
    model.zero_grad()
    losses = 0.0

    for idx,data in enumerate(train_iter):
        sources = data[0].to(device)
        targets = data[1].to(device)
        outputs, preds = model(sources, targets, scheudle_sampling('train',epoch))
           
        outputs = outputs[:,1:].reshape(-1, outputs.size(2))
        targets = targets[:,1:].reshape(-1)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad() 
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses += loss.item()
        
    return model, optimizer, losses / len(train_iter)

"""
Test model
"""

def test(model, test_loader, loss_function, process, device):
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    for idx,data in enumerate(test_loader):
        sources,targets = data[0].to(device),data[1].to(device)
        batch_size = sources.shape[0]
        outputs, preds = model(sources, targets, scheudle_sampling('val'))

        outputs = outputs[:,1:].reshape(-1, outputs.shape[2])
        targets = targets[:,1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        targets = targets.view(sources.shape[0], -1)
        preds = tokens_to_sequence(preds, process.cn_int2word)
        sources = tokens_to_sequence(sources, process.en_int2word)
        targets = tokens_to_sequence(targets, process.cn_int2word)
        
        bleu_score += compute_bleu(preds, targets)
        n += batch_size
    
    return loss_sum / len(test_loader), bleu_score / n

"""
Train process
"""

def train_process(config):
    process, en_embed, cn_embed =  build_data_process(['./cmn-eng/training.txt','./cmn-eng/validation.txt'],config)
    assert len(process.en_word2int) == en_embed.shape[0]
    assert len(process.cn_word2int) == cn_embed.shape[0]

    train_data = TextDataSet('./cmn-eng/training.txt', process, config.max_len)
    train_loader = DataLoader(train_data, batch_size = config.batch_size, shuffle = True, drop_last = False)

    val_data = TextDataSet('./cmn-eng/validation.txt', process, config.max_len)
    val_loader = DataLoader(val_data, batch_size = config.batch_size, shuffle = True)

    model, optimizer = build_model(config, en_embed, cn_embed)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_bleu = 99999,0
    for epoch in range(config.epoch_num):
        model, optimizer, train_loss = train(model, optimizer, train_loader, criterion, config.device, epoch+1)
        val_loss, bleu = test(model, val_loader, criterion, process, config.device)

        if bleu > best_bleu:
            best_loss, best_bleu = val_loss, bleu
            #save_model(model,config.store_model_path,epoch+1)
            print(f"Epoch num is {epoch+1}, Best val loss is {val_loss}, Best Bleu socre is {best_bleu}")
        else:
            print(f'Epoch num is {epoch+1} do nothing.')

    return model, criterion

"""
Test process
"""

def test_process(config, model, criterion):
    process, en_embed, cn_embed = build_data_process(['./cmn-eng/testing.txt'], config)
    assert len(process.en_word2int) == en_embed.shape[0]
    assert len(process.cn_word2int) == cn_embed.shape[0]

    data = TextDataSet('./cmn-eng/testing.txt', process, config.max_len)
    data_loader = DataLoader(data, batch_size = config.batch_size, shuffle = True, drop_last = False)

    test_loss, bleu_score = test(model, data_loader, criterion, process, config.device)

    return test_loss, bleu_score

def main():
    config = Config()
    model,criterion = train_process(config)
    test_loss, bleu_socre = test_process(config, model, criterion)
    print('val bleu ',bleu_socre)

if __name__ == '__main__':
    main()