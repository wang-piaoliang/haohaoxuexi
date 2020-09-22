# Preprocessing

数据预处理

**特征选择**

[https://www.cnblogs.com/pinard/p/9032759.html](https://www.cnblogs.com/pinard/p/9032759.html)

Filter, warpper, embedding\(l1, GBDT\)

一文读懂特征工程

[https://blog.csdn.net/v\_JULY\_v/article/details/81319999](https://blog.csdn.net/v_JULY_v/article/details/81319999)

归一化：将数值范围缩放到（0,1）,但没有改变数据分布的线性特征变换。

标准化：对数据的分布的进行转换，使其符合某种分布（比如正态分布）的一种非线性特征变换。

机器学习基础与实践（一）----数据清洗

[https://www.cnblogs.com/charlotte77/p/5606926.html](https://www.cnblogs.com/charlotte77/p/5606926.html)

**白化**的目的是去除输入数据的冗余信息。假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性

**概率模型不需要归一化**，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率

不需要归一化：决策树、rf

需要归一化：adaboost、gbdt、xgboost、svm、lr、KNN、Kmeans

归一化可提高梯度下降速度，有可能提高精度

数据分析中的缺失值处理

[https://blog.csdn.net/lujiandong1/article/details/52654703](https://blog.csdn.net/lujiandong1/article/details/52654703)

根据缺失原因有不同的处理

**TF-IDF**

字词的重要性随着它在文件中 出现的次数成正比增加，但同时会随着它在语料库中出现的频率成 反比下降。

TF: Term Frequency

**TF** = 词T在当前文中出现次数 / 文章的总词数

**IDF**：IDF\(t\) = log\(总文档数/ 含T的文档数+1\)

实际使用中，我们便经常用TF-IDF来计算权重，即TF-IDF = TF\(t\) \* IDF\(t\)

生动理解TF-IDF算法

[https://zhuanlan.zhihu.com/p/31197209](https://zhuanlan.zhihu.com/p/31197209)

特征工程之特征表达

[https://www.itcodemonkey.com/article/5689.html](https://www.itcodemonkey.com/article/5689.html)

![A screen shot of a map

Description automatically generated](../.gitbook/assets/0%20%281%29.png)

