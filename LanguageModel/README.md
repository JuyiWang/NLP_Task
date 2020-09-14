# 语言模型

## Language Model

语言模型:语言模型是对语句的概率分布的建模。 在给定序列的前提下,预测下一个词出现的概率:

$$P(w_i|w_1,...w_(i-1))$$

## 机器翻译

#### 数据集

英译汉: Chinese(Mandarin) - English (23610)

### 模型

#### Seq2seq with Attention

**Seq2seq** : Encoder(GRU)-Decoder(GRU)

**Attention** : Soft-Attention