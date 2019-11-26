# Machine Learning Engineer Nanodegree

## Capstone Proposal

张高超 2019.02.26

## Proposal

### Domain Background

\(Nature language process\) NLP是当前非常热门的主题，而比较句子相似度的任务是这一主题下很重要的话题，会被应用如问答，信息提取，文档相似度比较等各个方面。在NLP中有很多基础的模型，如词袋模型 \[1\]来表示一段文档的向量，TFIDF \[2\]来表示词频和词的重要程度，主题模型，如PCA \[3\]主成分分析来提取主题，LDA\[4\]主题模型等，这些技术提供了NLP发展的重要基础。其次在词向量上，有word2vec \[5\]，glove \[6\]，中文腾讯开源的语料库 \[7\]都给NLP分析数字化特征提取带来的重要的参考。 提取了有效的特征之后，在机器学习方面，也有很多相关的有效方法，如逻辑回归Logistic Regression，GBDT，RF，xgboost等等优秀的机器学习算法给NLP的处理提供了很多途径。 最近，随着深度学习的流行，深度学习网络模型在比较句子相似度上有了很大的发展，如DSSM \[8\], DeepMatch \[9\], CDSMM \[10\], ARC-I \[11\], CNTN \[12\], LSTM-RNN \[13\] and Bi-CNN-MI \[14\]。 在Quora这个项目中，提供了比较丰富的样本数据，是一个探索文本相似度方向的优秀数据，而且互联网内容大部分还是泛文本内容，对于以后的学习和扩展有很大帮助。

### Problem Statement

这个项目可以被看做是一个二分类问题，即两个问题是否相似，我们通过Qura dataset \[15\]下载数据，可以通过上面的机器学习或深度学习模型，训练出一个模型，使得f\(q1,q2\)映射到（0，1）的范围，这个数字越大，相似度越高，反之越低。其中包含比较复杂的情况，如句子的词很多相同但是逻辑上很大不同，或者词完全不同，但是是表示同一个意思，这些都需要我们测试不同的模型，找到最佳的模型来，在此项目使用log loss来作为损失函数，来测试模型的优劣。

### Datasets and Inputs

数据集是quora在Kaggle的比赛提供的，主要包括以下几个： id是每个问题对对应一个独立的id；qid1和qid2对应单个问题的独立id；question1和question2对应相应的具体问题内容；is\_duplicate对应question1和question2是否相同，相同为1，不同为0. 训练集有404290个问题对，相同的问题对所占的比例为36.92%，问题长度20-150；测试集有2345796个问题对，问题长度20-150。

### Solution Statement

为了解决这个问题，从两个方向去做：一个是机器学习，即先用特征工程提取有用的特征，如句子的长度，tfidf，关键词等等跟句子相似度有关的所有特征，之后选用不同的特征组合去测试不同的机器学习算法如LR，GBDT，RF等等，使用bagging或者boosting进行集成学习；第二是深度学习方法，首先用不同的语料库词向量来编码每一个问题，把每一个问题变为固定长度的特征向量，然后尝试不同的神经网络模型，或者用不同的比较相似度的方法去尝试。

### Benchmark Model

为了有benchmark model，我们可以设定一个0到1的阈值来定义了两个句子之间的相似度，quora自己曾使用过随机森林模型通过使用各种各样的特征如word2vec词向量，余弦相似度等等，在kaggle上也会有基准的评判标准，根据项目要求提交分数需要达到kaggle private leaderboard 的top 20%,对于该题目的就是660th/3307,对应logloss得分为0.18267。

### Evaluation Metrics

在这个项目中我使用Kaggle比赛中使用的评价函数，即log loss。每一对问题对都会有一个0到1的相似度概率，具体的公式可以详见wiki \[16\]。

### Project Design

第一步数据分析，对问题的各种特征有一个总体的统计和认识； 第二步数据预处理，比如问题量化后的一些统计学特征，还有一些语义上的分析； 第三步特征工程，使用自然语言处理方法抽取特征，如长度，词向量，tfidf等等，统计这些特征的分布并进行可视化; 第四步训练模型，使用两到三个模型进行组合训练，如LR，RF，GBDT等等，对于不同的组合进行调参和集成学习的训练; 第五步（可选）探索，了解和尝试最新的NLP神经网络模型如DSSM和BERT \[17\]等等。

### References

\[1\] [https://zh.wikipedia.org/wiki/%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B](https://zh.wikipedia.org/wiki/%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8B)  
\[2\] [https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)  
\[3\] [https://zh.wikipedia.org/zh-tw/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90](https://zh.wikipedia.org/zh-tw/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)  
\[4\] [https://en.wikipedia.org/wiki/Latent\_Dirichlet\_allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)  
\[5\] [https://en.wikipedia.org/wiki/Word2vec](https://en.wikipedia.org/wiki/Word2vec)  
\[6\] Jeffrey Pennington, Richard Socher, Christopher D. Manning; GloVe: Global Vectors for Word Representation.  
\[7\] [https://ai.tencent.com/ailab/nlp/embedding.html](https://ai.tencent.com/ailab/nlp/embedding.html)  
\[8\] Huang, P.-S.; He, X.; Gao, J.; Deng, L.; Acero, A.; and Heck, L. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management \(CIKM\) , 2333– 2338.  
\[9\] Lu, Z., and Li, H. 2013. A Deep Architecture for Matching Short Texts. In Advances in Neural Information Processing Systems \(NIPS\) , 1367–1375.  
\[10\] Shen, Y.; He, X.; Gao, J.; Deng, L.; and Mesnil, G. 2014. A Latent Semantic Model with Convolutional-Pooling Structure for Information Retrieval. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management \(CIKM\) , 101–110.  
\[11\] Hu, B.; Lu, Z.; Li, H.; and Chen, Q. 2014. Convolutional neural network architectures for matching natural language sentences. In Advances in Neural Information Processing Systems \(NIPS\) , 2042–2050.  
\[12\] Qiu, X., and Huang, X. 2015. Convolutional Neural Tensor Network Architecture for Community-Based Question Answering. In Proceedings of the 24th International Joint Conference on Artificial Intelligence \(IJCAI\) , 1305–1311.  
\[13\] Palangi, H.; Deng, L.; Shen, Y.; Gao, J.; He, X.; Chen, J.; Song, X.; and Ward, R. K. 2015. Deep sentence embedding using the long short term memory network: Analysis and application to information retrieval. CoRR abs/1502.06922.  
\[14\] Yin, W., and Schutze, H. 2015a. Convolutional Neural Network for Paraphrase Identifica- tion. In The 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies \(NAACL\) , 901–911.  
\[15\] Quora Question Pairs - Can you identify question pairs that have the same intent? [https://www.kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs)  
\[16\] [https://en.wikipedia.org/wiki/Cross\_entropy](https://en.wikipedia.org/wiki/Cross_entropy)  
\[17\] Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova; BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 \[cs.CL\].

