# Papers-of-aspect-level-sentiment-classification
## ALSC Method
### Traditional Method

1. [Mining and Summarizing Customer Reviews](https://dl.acm.org/doi/abs/10.1145/1014052.1014073)
2. [Sentiment analysis using support vector machines with diverse information sources](https://aclanthology.org/W04-3253.pdf)
3. [Semi-Supervised Polarity Lexicon Induction](https://aclanthology.org/E09-1077.pdf)
4. [Target-dependent Twitter Sentiment Classification](https://aclanthology.org/P11-1016.pdf)
5. [NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets](https://arxiv.org/abs/1308.6242)
6. [NRC-Canada-2014: Detecting Aspects and Sentiment in Customer Reviews](https://aclanthology.org/S14-2076.pdf)

### Deep Learning Method
#### Early Deep Learning Method

1. [Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](https://aclanthology.org/P14-2009.pdf)
2. [Effective LSTMs for Target-Dependent Sentiment Classification](https://arxiv.org/abs/1512.01100)

#### Attention-based Method

1. [Attention-based LSTM for Aspect-level Sentiment Classification](https://aclanthology.org/D16-1058.pdf)
2. [Aspect Level Sentiment Classification with Deep Memory Network](https://arxiv.org/abs/1605.08900)
3. [Interactive Attention Networks for Aspect-Level Sentiment Classification](https://arxiv.org/abs/1709.00893)
4. [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](https://aclanthology.org/D17-1047.pdf)
5. [Multi-grained Attention Network for Aspect-Level Sentiment Classification](https://aclanthology.org/D18-1380/?ref=https://githubhelp.com)
6. [Learning to Attend via Word-Aspect Associative Fusion for Aspect-Based Sentiment Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/12049)
7. [Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks](https://arxiv.org/pdf/1804.06536.pdf?ref=https://githubhelp.com)
8. [Attentional Encoder Network for Targeted Sentiment Classification](https://arxiv.org/abs/1902.09314)
9. [Earlier Attention? Aspect-Aware LSTM for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/1905.07719)

#### CNN-based Method

1. [Aspect Based Sentiment Analysis with Gated Convolutional Networks](https://arxiv.org/abs/1805.07043)
2. [Transformation Networks for Target-Oriented Sentiment Classification](https://arxiv.org/abs/1805.01086)
#### Dependency-based but not GNN
1. [Adaptive recursive neural network for target-dependent twitter sentiment classification](https://aclanthology.org/P14-2009.pdf)
2. [Effective Attention Modeling for Aspect-Level Sentiment Classification](https://ieeexplore.ieee.org/abstract/document/8573324)
3. [Syntax-Aware Aspect-Level Sentiment Classification with Proximity-Weighted Convolution Network](https://dl.acm.org/doi/abs/10.1145/3331184.3331351)
4. [Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis](https://aclanthology.org/2020.acl-main.293/?ref=https://githubhelp.com)
#### GNN-based Method

1. [Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa](https://arxiv.org/abs/2104.04986)(可以当做综述来看)
2. [Aspect-based Sentiment Classification with Aspect-specific Graph Convolutional Networks](https://arxiv.org/abs/1909.03477)
3. [Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://aclanthology.org/D19-1569/?ref=https://githubhelp.com)
4. [Syntax-Aware Aspect Level Sentiment Classification with Graph Attention Networks](https://arxiv.org/abs/1909.02606)
5. [Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2004.12362)
6. [Convolution over Hierarchical Syntactic and Lexical Graphs for Aspect Level Sentiment Analysis](https://aclanthology.org/2020.emnlp-main.286/)
7. [Inducing Target-Specific Latent Structures for Aspect Sentiment Classification](https://aclanthology.org/2020.emnlp-main.451/)
8. [Jointly Learning Aspect-Focused and Inter-Aspect Relations with Graph Convolutional Networks for Aspect Sentiment Analysis](https://aclanthology.org/2020.coling-main.13/)
9. [Dual Graph Convolutional Networks for Aspect-based Sentiment Analysis](https://aclanthology.org/2021.acl-long.494/)

## Tools about ALSC
### Datasets

1. Laptop和restaurant数据集：[SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/)

2. Twitter数据集：[Adaptive recursive neural network for target-dependent twitter sentiment classification](https://aclanthology.org/P14-2009.pdf)

3. MAMS数据集：[A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://aclanthology.org/D19-1654.pdf)


### Word Vector
1. [A Neural Probabilistic Language Model](https://proceedings.neurips.cc/paper/2000/hash/728f206c2a01bf572b5940d7d9a8fa4c-Abstract.html)
2. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
3. [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)
4. [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf)
5. [Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.636.1284&rep=rep1&type=pdf)
### RNN

1. LSTM：[Long short-term memory](https://ieeexplore.ieee.org/abstract/document/6795963/)
2. GRU：[Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

### GNN

1. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
2. [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/)
3. [GRAPH ATTENTION NETWORKS](https://arxiv.org/abs/1710.10903)

### Dependency Parser

1. [A fast and accurate dependency parser using neural networks](https://aclanthology.org/D14-1082.pdf)

2. [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)

3. [Rethinking Self-Attention: An Interpretable Self-Attentive Encoder-Decoder Parser](https://openreview.net/forum?id=59St7jseQ-)

### BERT
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

### Others

1. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
2. Glorot初始化：[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a)
3. ResNet：[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
4. LSR标签平滑化的操作：[Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)

