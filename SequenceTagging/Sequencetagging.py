# -*- coding: utf-8 -*-

"""
SequenceTagging.py

- Sequence Tagging Task :
    - Embedding :
        - Word2vec Embedding
        - Character Embedding
    - Model :
        - BILSTM-CNN-CRF
"""

import os
import time
import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from gensim.models import word2vec
from torch.utils.data import Dataset,DataLoader

"""
Data loader
"""

class CoNNL2003DataLoader():   
    def read_train_dev_test(self, config):
        word_train, tag_train = self.read_data(config.train_path)
        word_dev, tag_dev = self.read_data(config.valid_path)
        word_test, tag_test = self.read_data(config.test_path)
        return word_train, tag_train, word_dev, tag_dev, word_test, tag_test

    def read_data(self, path):
        word_sequence = list()
        tag_sequence = list()
        with open(path, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
        
        curr_word = list()
        curr_tag = list()
        for i in range(len(lines)):
            line = lines[i].strip('\n').split('\t')
            if len(line) == 1:
                assert len(curr_word) == len(curr_tag), 'tag doesn\'t match word'
                word_sequence.append(curr_word)
                tag_sequence.append(curr_tag)
                curr_word = list()
                curr_tag = list()
                continue
            curr_word.append(line[0])
            curr_tag.append(line[1])
            if i == len(lines)-1:
                word_sequence.append(curr_word)
                tag_sequence.append(curr_tag)
        assert len(word_sequence) == len(tag_sequence), 'samples doesn\'t match tags'
        print(f'Load from {path} : {len(word_sequence)} samples')
        return word_sequence,tag_sequence

"""
Tokenizer
"""

class Tokenizer():
    def __init__(self, config):
        self.embed_dim = config.word_embed_dim
        self.embed_typ = config.word_embed_type
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word2vec = {}

    def add_embedding(self, word):
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        vector = torch.empty(1, self.embed_dim)
        if word != '<PAD>':
            torch.nn.init.xavier_normal_(vector)
        self.idx += 1
        return vector

    def train_word2vec(self, data, window = 5):
        word2vec_model = word2vec.Word2Vec(data, size = self.embed_dim, window = window, iter = 10)
        self.word2vec = word2vec_model.wv.vocab
        self.w2v = word2vec_model
    
    def load_glove(self, path):
        num = -1 * self.embed_dim
        with open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
            for line in f:
                tokens = line.split()
                self.word2vec[' '.join(tokens[:num])] = np.asarray(tokens[num:], dtype = 'float32')
    
    def build_embedding(self):
        pad_embed = self.add_embedding('<PAD>')
        embedding = []
        for word in self.word2vec:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            embedding.append(self.word2vec[word] if self.embed_typ is 'glove' else self.w2v.wv[word])
        unk_embed = self.add_embedding('<UNK>')
        embedding = torch.tensor(embedding)
        embedding = torch.cat((pad_embed, embedding), 0)
        self.embedding_matrix = torch.cat((embedding, unk_embed), 0)
        assert self.embedding_matrix.shape[0] == self.idx, 'Word embedding index do not match'
    
    def build_embed_matrix(self, data = None, path = None):
        if self.embed_typ == 'w2v':
            self.train_word2vec(data)
        else:
            self.load_glove(path)
        self.build_embedding()
        print(f'Embed matrix load successfully')
        return self.idx, self.embedding_matrix
    
    def conver_tokens_to_ids(self, sequence):
        seq = [self.word2idx[word] if word in self.word2idx else self.word2idx['<UNK>'] for word in sequence]
        return np.asarray(seq, dtype = 'int64')

"""
Character tokenizer
"""

class CharacterTokenizer():
    def __init__(self, config):
        self.max_len = config.max_word_len
        self.embed_dim = config.char_embed_dim
        self.root = config.root_path
        self.char2idx = {}
        self.idx = 1
        
    def _init_dict(self):
        for item in list(string.ascii_letters):
            self.char2idx[item] = self.idx
            self.idx += 1
        for item in list(string.digits):
            self.char2idx[item] = self.idx
            self.idx += 1

    def _check_dict(self, data):
        for seq in data:
            for word in seq:
                for char in word:
                    if char not in self.char2idx :
                        self.char2idx[char] = self.idx
                        self.idx += 1
        assert len(self.char2idx)+1 == self.idx ,'Char embedding index do not match'

    def build_char_embedding(self):
        embedding = torch.empty(self.idx, self.embed_dim)
        torch.nn.init.uniform_(embedding[1:,:], a = -1 * math.sqrt(3 / self.embed_dim), b = math.sqrt(3 / self.embed_dim))
        return self.idx, embedding
    
    def conver_chars_to_ids(self, seq):
        word_list = list()
        for word in seq:
            char_list = [0]*self.max_len
            char_ = [self.char2idx[char] for char in word]
            char_ = char_[:self.max_len]
            char_list[:len(char_)] = char_
            word_list.append(char_list)
        return np.asarray(word_list, dtype = 'int64')

"""
Dataset
"""

class NerDataset(Dataset):
    def __init__(self, word_tokenizer, char_tokenizer, sequence, tag):
        self.tag_dic = {}
        self.X, self.Y = [],[]
        for seq in sequence:
            word_seq = word_tokenizer.conver_tokens_to_ids(seq)
            tag_seq = char_tokenizer.conver_chars_to_ids(seq)
            self.X.append((word_seq, tag_seq))
        self._init_tag_dic(tag)
        for label in tag:
            y = [self.tag_dic[item] for item in label]
            self.Y.append(y)
        assert len(self.X) == len(self.Y), 'Data and target not match'

    def __getitem__(self,index):
        return self.X[index],self.Y[index]

    def __len__(self):
        return len(self.X)

    def _init_tag_dic(self, tags):
        self.tag_dic = {
            'B-ORG': 1, 'I-ORG': 2, 
            'B-MISC': 3, 'I-MISC': 4, 
            'B-PER': 5, 'I-PER': 6, 
            'B-LOC': 7, 'I-LOC': 8, 
            'O': 9,
            '<START>':0, '<STOP>':10
        }

"""
Character embedding
"""

class CharEmbeddingCNN(nn.Module):
    def __init__(self, config, embed_matrix):
        super(CharEmbeddingCNN, self).__init__()
        self.word_len = config.max_word_len
        self.embed_dim = config.char_embed_dim
        self.filter_num = config.filter_num
        self.char_embedding = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype = torch.float))
        self.dropout = nn.Dropout(config.dropout)
        self.char_cnn = nn.Conv1d(self.embed_dim, self.filter_num, kernel_size = 3, stride = 1, padding = 1)
        self.max_pool = nn.MaxPool1d(kernel_size = self.word_len)

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        batch_size = inputs.shape[0]
        embedding = self.dropout(self.char_embedding(inputs))
        out = torch.empty((batch_size, seq_len, self.filter_num))
        for i in range(batch_size):
            i_embedding = embedding[i,:,:,:]   
            cnn_out = self.char_cnn(i_embedding.permute(0,2,1).contiguous())
            max_out = self.max_pool(cnn_out).squeeze(2)
            out[i,:,:] = max_out
        return out

"""
BILSTM-CRF
"""

class BILSTM_CRF(nn.Module):
    def __init__(self, config, tag_to_idx):
        super(BILSTM_CRF, self).__init__()
        self.target_size = len(tag_to_idx)
        self.embed_dim = config.char_embed_dim + config.word_embed_dim
        self.hid_dim = config.hid_dim
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(self.embed_dim, self.hid_dim, num_layers = self.num_layers, batch_first = True, bidirectional = True)
        self.target_size = len(tag_to_idx)
        self.tag_to_idx = tag_to_idx
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hid_dim * 2, self.target_size)
        )
        self._init_para()
        # transition from j to i : transition[i,j]
        self.transition = nn.Parameter(torch.randn(self.target_size, self.target_size)) 
        self.transition.data[self.tag_to_idx['<START>'],:] = -10000
        self.transition.data[:, self.tag_to_idx['<STOP>']] = -10000

    def _init_para(self):
        nn.init.uniform_(self.dense[1].weight.data, a = -(6 / (self.hid_dim*2 + self.target_size))**0.5, b = (6 / (self.hid_dim*2 + self.target_size))**0.5 )
        nn.init.constant_(self.dense[1].bias.data, 0)

    def _forward_alg(self, feats):
        # init_alphas : [1, target_size] , <START> = 0 others -10000
        init_alphas = torch.full((1,self.target_size), -10000.)  
        init_alphas[0][self.tag_to_idx['<START>']] = 0
        # forward_var : [1, target_size]
        forward_var = init_alphas   
        # feat : all the label of word_i
        for feat in feats:  # feats : [seq_len, target_size] | feats : [1, target_size]  
            alphs_t = []
            # next_tag : the j_th label of word_i
            for next_tag in range(self.target_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1,-1).expand(1, self.target_size)  # emit_score : [1, target_size]
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                tran_socre = self.transition[next_tag]
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + emit_score + tran_socre
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphs_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphs_t).view(1,-1)     # forward_var : [1, target_size]
        terminal_var = forward_var + self.transition[self.tag_to_idx['<STOP>']]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, target):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx['<START>']], dtype = torch.long), torch.LongTensor(target)])
        for i, feat in enumerate(feats):
            score = score + self.transition[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transition[self.tag_to_idx['<STOP>'], tags[-1]]
        return score

    def _neg_log_likelihood(self, sentence, tags):
        feats,_ = self.lstm(sentence)
        feats = self.dense(feats)
        feats = feats.squeeze(0)
        forward_socre = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_socre - gold_score

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.target_size), -10000.)
        init_vvars[0][self.tag_to_idx['<START>']] = 0
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # holds the backpointers for this step
            bptrs_t = []
            # holds the viterbi variables for this step
            viterbivars_t = []
            for next_tag in range(self.target_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transition[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1,-1)
            backpointers.append(bptrs_t)
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transition[self.tag_to_idx['<STOP>']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_idx['<START>']  
        best_path.reverse()
        return path_score, best_path

    def forward(self,sentence):
        feats,_ = self.lstm(sentence)
        feats = self.dense(feats)
        feats = feats.squeeze(0)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq

"""
BILSTM-CNN-CRF
"""

class BILSTM_CNN_CRF(nn.Module):
    def __init__(self, config, char_embedding, bilstm_crf, word_embed_matrix):
        super(BILSTM_CNN_CRF, self).__init__()
        self.char_embed = char_embedding
        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(word_embed_matrix, dtype = torch.float))
        self.bilstm_crf = bilstm_crf
        self.dropout = nn.Dropout(config.dropout)

    def _cat_word_char_embedding(self, sentence, word_seq):
        char_embed = self.char_embed(word_seq)
        word_embed = self.word_embed(sentence)
        embedding = torch.cat((char_embed, word_embed), -1)
        embedding = self.dropout(embedding)
        return embedding
    
    def _neg_loss(self, inputs, tags):
        embedding = self._cat_word_char_embedding(inputs[0],inputs[1])
        sub_score = self.bilstm_crf._neg_log_likelihood(embedding, tags)
        return sub_score
    
    def forward(self, inputs):
        embedding = self._cat_word_char_embedding(inputs[0], inputs[1])
        score, tag_seq = self.bilstm_crf.forward(embedding)
        return score, tag_seq

"""
Build model
"""

def build_model(config, char_embed_matrix, tag_to_idx, word_embed_matrix):
    char_embedding = CharEmbeddingCNN(config, char_embed_matrix)
    bilstm_crf = BILSTM_CRF(config, tag_to_idx)
    bilstm_cnn_crf = BILSTM_CNN_CRF(config, char_embedding, bilstm_crf, word_embed_matrix)
    return bilstm_cnn_crf

"""
Configuration
"""

class Configuration():
    def __init__(self,):
        # Data file
        self.root_path = './SequenceTagging'
        self.train_path = './data/train.txt'
        self.valid_path = './data/valid.txt'
        self.test_path = './data/test.txt'
        # Word embedding
        self.filter_num = 30
        self.max_word_len = 8
        self.char_embed_dim= 30
        self.word_embed_dim = 100
        self.word_embed_type = 'glove'
        # BILSTM-CRF
        self.hid_dim = 128
        self.num_layers = 1
        # Train process
        self.batch_size = 1
        self.lr = 1e-4
        self.dropout = 0.5
        self.epoch_num = 10
        self.test = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Utils
"""

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

tags = [(1,2),(3,4),(5,6),(7,8)]

def _find_tag(labels, B_label = 1, I_label = 2):
    result = []
    for num in range(len(labels)):
        if labels[num] == B_label:
            song_pos = num
            length = 1
            for num2 in range(num, len(labels)):
                if labels[num2] == I_label:
                    length += 1
                else:
                    result.append((song_pos, length))
                    break
            num = num2
    return result

def find_all_tag(labels):
    result = {}
    for tag in tags:
        res = _find_tag(labels, B_label = tag[0], I_label = tag[1])
        result[tag[0]] = res
    return result

def precision(pre_labels, true_labels):
    pre = []
    pre_result = find_all_tag(pre_labels)
    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    return sum(pre)/len(pre) if len(pre) != 0 else 0

def recall(pre_labels, true_labels):
    re = []   
    true_result = find_all_tag(true_labels)
    for name in true_result:
        for x in true_result[name]:
            if x:
                if pre_labels[x[0]:x[0]+x[1]] == true_labels[x[0]:x[0]+x[1]]:
                    re.append(1)
                else:
                    re.append(0)
    return sum(re)/len(re) if len(re) != 0 else 0

def compute_f1_score(pre, true):
    prec = precision(pre, true)
    reca = recall(pre, true)
    return (2 * prec * reca)/(prec + reca) if (prec+reca) != 0 else 0

"""
Main
"""

if __name__ == '__main__':
    config = Configuration()
    dataloader = CoNNL2003DataLoader()
    train_data,train_tag,valid_data,valid_tag,test_data,test_tag = dataloader.read_train_dev_test(config)

    char_tokenizer = CharacterTokenizer(config)
    char_tokenizer._init_dict()
    char_tokenizer._check_dict(train_data)
    char_vocab_size, char_embed_matrix = char_tokenizer.build_char_embedding()

    word_tokenizer = Tokenizer(config)
    word_vocab_size, word_embed_matrix = word_tokenizer.build_embed_matrix(path = './glove.6B.100d.txt')

    train_dataset = NerDataset(word_tokenizer, char_tokenizer, train_data, train_tag)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1,shuffle = True)
    tag_to_idx = train_dataset.tag_dic

    valid_dataset = NerDataset(word_tokenizer, char_tokenizer, valid_data, valid_tag)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1,shuffle = True)
   
    model = build_model(config, char_embed_matrix, tag_to_idx, word_embed_matrix)

    optimz = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 1e-4) 
    
    for idx in range(config.epoch_num):
        loss_list = []
        begin_time = time.time()
        for _,(X,Y) in enumerate(train_data_loader): 
            optimz.zero_grad()
            loss = model._neg_loss(X,Y)
            loss_list.append(loss.item())
            loss.backward()
            optimz.step()
        now_time = time.time()
        mean_loss = np.mean(loss_list)
        print(f'Epoch num is {idx+1}, use time {now_time - begin_time},loss is {mean_loss}')
        begin_time = now_time

        F1_list = []
        all_pre = []
        all_res = []
        for _,(X,Y) in enumerate(train_data_loader):
            score,pre = model.forward(X)
            all_pre = all_pre + pre
            res = [int(item) for item in Y]
            all_res = all_res + res
        f1 = compute_f1_score(all_pre, all_res)
        F1_list.append(f1)
        print('Train F1 :','%.3f'%np.mean(F1_list),'%.3f'%np.max(F1_list)) 

        F1_list = []
        all_pre = []
        all_res = []
        for _,(X,Y) in enumerate(valid_data_loader):
            score, pre = model.forward(X)
            all_pre = all_pre + pre
            res = [int(item) for item in Y]
            all_res = all_res + res
        f1 = compute_f1_score(all_pre, all_res)
        F1_list.append(f1)
        print('Test F1 :','%.3f'%np.mean(F1_list),'%.3f'%np.max(F1_list))