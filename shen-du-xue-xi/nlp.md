# NLP

NLP领域任务

[https://juejin.im/post/5d5bdb4951882536d218a237\#heading-1](https://juejin.im/post/5d5bdb4951882536d218a237#heading-1)

**Word2Vec**

\[NLP\] 秒懂词向量Word2vec的本质

[https://zhuanlan.zhihu.com/p/26306795](https://zhuanlan.zhihu.com/p/26306795)

如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做『**Skip-gram 模型**』

而如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 『**CBOW 模型**』

**一般来说，cbow模型快一些，skip-gram模型效果好一些**

![A picture containing text, map

Description automatically generated](../.gitbook/assets/0.jpeg)

理解 Word2Vec 之 Skip-Gram 模型

[https://zhuanlan.zhihu.com/p/27234078](https://zhuanlan.zhihu.com/p/27234078)

word2vec算出的词向量怎么衡量好坏？

[https://www.zhihu.com/question/37489735](https://www.zhihu.com/question/37489735)

下游模型表现，wordanalogy，similarity，visualization

无监督学习，所以和其他无监督模型一样，难以直接评价。比较常用的做法是，把它丢到另一个有监督的系统（任务）中去当做特征，看看对这个有监督系统有多大改善。

**hierarchical softmax**: 本质是把 N 分类问题变成 log\(N\)次二分类

**negative sampling**:本质是预测总体类别的一个子集

通俗易懂理解 - 负采样

[https://zhuanlan.zhihu.com/p/39684349](https://zhuanlan.zhihu.com/p/39684349)

更新positive word和随机选择一小部分的negative words（比如选5个negative words）来更新对应的权重。

![Important](../.gitbook/assets/1%20%281%29.png) 从word2vec、glove、ELMo到BERT

[https://www.cnblogs.com/rucwxb/p/10277217.html](https://www.cnblogs.com/rucwxb/p/10277217.html)

NLP任务分成两部分，预训练产生词向量，对词向量操作（下游具体NLP任务）。

从word2vec到ELMo到BERT，做的其实主要是把下游具体NLP任务的活逐渐移到预训练产生词向量上。

**Word2Vec**

![NO 
CBOW/Skip-Gram 
next word 
Yes 
word-level ](../.gitbook/assets/2%20%282%29.png)

**Glove**: word2vec只考虑到了词的局部信息，没有考虑到词与局部窗口外词的联系

**ElMo**: static向量变成上下文相关的dynamic向量，将encoder操作转移到预训练产生词向量过程实现。

![Machine generated alternative text:
&#x9884;&#x8BAD;&#x7EC3;enc&#x3002;ding&#xFF08;&#x4E0A;&#x4E0B;&#x6587;&#x76F8;&#x5173;&#xFF09;
Yes
&#x6A21;&#x578B;
bi-lstm
&#x987A;&#x6D4B;&#x76EE;&#x6807;
nextword
&#x4E0B;&#x6E38;&#x5177;&#x4F53;&#x4EFB;&#x52A1;
&#x9700;&#x8981;&#x8BBE;&#x7F6E;&#x6BCF;&#x5C42;&#x53C2;&#x6570;
&#x8D1F;&#x91C7;&#x6837;
No
&#x7EA7;&#x522B;
word-level](../.gitbook/assets/3%20%283%29.png)

**BERT**

工作方式跟ELMo是类似的，但ELMo的语言模型使用的是LSTM问题有两个：

* * **单方向**，即使是BiLSTM双向模型，也只是在loss处做一个简单的相加，没法同时考虑前后文。
  * 序列模型**并行**计算的能力差。

训练出的word-level向量变成sentence-level的向量，下游具体NLP任务调用更方便

![Yes 
Transformer 
masked 
HEMLP 
Yes 
sentence-level ](../.gitbook/assets/4%20%284%29.png)

词向量经典模型：从word2vec、glove、ELMo到BERT

[https://zhuanlan.zhihu.com/p/51682879](https://zhuanlan.zhihu.com/p/51682879)

**BERT**

三个亮点: Masked LM、transformer、sentence-level

用整篇文章85%的词预测\[Mask\]住的15%的词。15%被盖住的词中：

80%的词就用\[mask\]符号盖住; 10%的词保留原来真实的词; 10%的词用随机的一个词替代

BERT：L=24，H=1024，A=16

![Important](../.gitbook/assets/5%20%282%29.png) Transformer图解

[http://fancyerii.github.io/2019/03/09/transformer-illustrated/](http://fancyerii.github.io/2019/03/09/transformer-illustrated/)

![A screenshot of a cell phone

Description automatically generated](../.gitbook/assets/6%20%282%29.png)

![A screenshot of a cell phone

Description automatically generated](../.gitbook/assets/7.png)

**Attention**

通过参数，来控制每一个词在语义向量中的权重，从而提升最终效果

AM模型理解成影响力模型也是合理的，就是说生成目标单词的时候，输入句子每个单词对于生成这个单词有多大的影响程度

![A close up of a clock

Description automatically generated](../.gitbook/assets/8.jpeg)

自然语言处理中的Attention Model：是什么及为什么

[https://blog.csdn.net/malefactor/article/details/50550211](https://blog.csdn.net/malefactor/article/details/50550211)

Attention Model（注意力模型）学习总结

[https://www.cnblogs.com/guoyaohua/p/9429924.html](https://www.cnblogs.com/guoyaohua/p/9429924.html)

目前主流的attention方法都有哪些？ - JayLou的回答 - 知乎

[https://www.zhihu.com/question/68482809/answer/597944559](https://www.zhihu.com/question/68482809/answer/597944559)

**Attention种类**

* * Spatial Attention 空间注意力和Temporal Attention 时间注意力。
  * Soft Attention和Hard Attention。
    1. Soft Attention是所有的数据都会注意，都会计算出相应的注意力权值，不会设置筛选条件。
    2. Hard Attention会在生成注意力权重后筛选掉一部分不符合条件的注意力，让它的注意力权值为0，即可以理解为不再注意这些不符合条件的部分。

**注意力权重获取**的过程

* * 第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，cosine，MLP等；
  * 第二步一般是使用一个softmax函数对这些权重进行归一化；
  * 最后将权重和相应的键值value进行加权求和得到最后的attention。

attention各种形式总结 [link](https://blog.csdn.net/qq_41058526/article/details/80783925)

**Self Attention**: Attention机制发生在Target的元素和Source中的所有元素之间。而Self Attention顾名思义，指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制。

**Self Attention与RNN/LSTM**

引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的**信息累积才能**将两者联系起来，而距离越远，有效捕获的可能性越小。

但是Self Attention在计算过程中会**直接**将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。

**RNN**

如何深度理解RNN？

[https://zhuanlan.zhihu.com/p/45289691](https://zhuanlan.zhihu.com/p/45289691)

假设RNN的输入是一句话，这句话中有多个单词，那么RNN需要forward多次

* * RNN的结构与全连接网络基本一致
  * RNN具有时间展开的特点，这是由其输入决定的
  * 全连接网络对一个样本做一次forward，RNN对一个样本做多次forward

![rnn\_arch](../.gitbook/assets/9%20%281%29.png)

\[译\]理解 LSTM 网络

[https://yugnaynehc.github.io/2017/01/03/understanding-lstm-networks/](https://yugnaynehc.github.io/2017/01/03/understanding-lstm-networks/)

在一种叫做门\(gate\)的结构的精心控制下，LSTM 具有往细胞状态中添加或者移除信息的能力。

* * 决定需要从之前的细胞状态中剔除多少信息。
  * 确定哪些新信息需要被存储到细胞状态中
  * 确定将会输出什么内容

![RNN &#x7684;&#x5FAA;&#x73AF;&#x6A21;&#x5757;&#x5177;&#x6709;&#x4E00;&#x4E2A;&#x5C42;&#x3002;](../.gitbook/assets/10%20%283%29.png)

![LSTM &#x7684;&#x5FAA;&#x73AF;&#x6A21;&#x5757;&#x5177;&#x6709;&#x56DB;&#x4E2A;&#x4EA4;&#x4E92;&#x5C42;&#x3002;](../.gitbook/assets/11%20%282%29.png)

LSTM的公式推导详解

[https://blog.csdn.net/u010754290/article/details/47167979](https://blog.csdn.net/u010754290/article/details/47167979)

**FastText**

FastText算法原理解析（对比w2v）

[https://www.cnblogs.com/huangyc/p/9768872.html](https://www.cnblogs.com/huangyc/p/9768872.html)

fastText原理及实践

[https://zhuanlan.zhihu.com/p/32965521](https://zhuanlan.zhihu.com/p/32965521)

优点：fastText（浅层网络）往往能取得和深度网络相媲美的精度，却在训练时间上比深度网络快许多数量级。

优势：模型简单, 训练速度快，效果不错。

缺点：模型可解释型不强，在调优模型的时候，很难根据训练的结果去针对性的调整具体的特征，因为在textCNN中没有类似gbdt模型中特征重要度\(feature importance\)的概念, 所以很难去评估每个特征的重要度。

**TextCNN**

![Important](../.gitbook/assets/12%20%283%29.png) 文本分类算法TextCNN原理详解

[https://www.cnblogs.com/ModifyRong/p/11319301.html](https://www.cnblogs.com/ModifyRong/p/11319301.html)

听说你不会调参？TextCNN的优化经验Tricks汇总

[https://www.cnblogs.com/ModifyRong/p/11442661.html](https://www.cnblogs.com/ModifyRong/p/11442661.html)

![A screenshot of a cell phone

Description automatically generated](../.gitbook/assets/13%20%282%29.png)

TextCNN 其实只有一层卷积,一层max-pooling, 最后将输出外接softmax来n分类。

基于 Tensorflow 的 TextCNN 在搜狗新闻数据的文本分类实践

[https://www.libinx.com/2018/text-classification-cnn-by-tensorflow/](https://www.libinx.com/2018/text-classification-cnn-by-tensorflow/)

![1 
3 
4 
8 
9 
13 
14 
18 
19 
class TCNNConfig\(object\) : 
&quot; &quot; &quot;CNN&#xB1;dASZf&quot; &quot; &quot; 
embeddi ng\_dim 64 \# 
seq\_tength - 
num\_classes 
num filters 
ke rnel\_si ze 
vocab\_size - 
hidden\_dim = 
= 11 \# 
= 256 
128 \# 
learning\_rate = le-3 \# 
batch size = 64 
num\_epochs - 
16 
print\_per\_batch - 
save\_per\_batch = 10 \# ](../.gitbook/assets/14%20%283%29.png)

![with 
\# CNN layer 
tf. layers. convld \(embeddi ng\_inputs, self. config. num\_filters, 
conv - 
\# global max pooling layer 
gmp = tf.reduce\_max\(conv, , name= &apos; gmp&apos;\) 
self. config. kernel\_size, 
- &apos; conv&apos;\) 
name- ](../.gitbook/assets/15%20%282%29.png)

TextCNN文本分类（keras实现）

[https://blog.csdn.net/asialee\_bird/article/details/88813385](https://blog.csdn.net/asialee_bird/article/details/88813385)

FAISS 用法

[https://zhuanlan.zhihu.com/p/40236865](https://zhuanlan.zhihu.com/p/40236865)

