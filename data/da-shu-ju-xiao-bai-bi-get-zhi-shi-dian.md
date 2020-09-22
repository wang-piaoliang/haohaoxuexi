# 大数据小白必Get知识点！

分布式计算框架要解决两个问题：如何分发数据和如何分发计算。

Hadoop 使用 HDFS 来解决分布式数据问题，MapReduce 计算范式提供有效的分布式计算。

## **Spark** <a id="Spark"></a>

可以理解为在Hadoop MapReduce的一种改进，MapReduce是面向磁盘的，**Spark是面向内存的**。这使得Spark能够为多个不同数据源的数据提供近乎实时的处理性能。若在内存中运行，Spark要比MapReduce快100倍。

### Spark组成 <a id="Spark%E7%BB%84%E6%88%90"></a>

**Spark Core：**实现了 Spark 的基本功能，包含任务调度、内存管理、错误恢复、与存储系统 交互等模块。Spark Core 中还包含了对弹性分布式数据集\(resilient distributed dataset，简称RDD\)的 API 定义。

**Spark SQL：**是 Spark 用来操作结构化数据的程序包。通过 Spark SQL，我们可以使用 SQL 或者 Apache Hive 版本的 SQL 方言\(HQL\)来查询数据。Spark SQL 支持多种数据源，比 如 Hive 表、Parquet 以及 JSON 等。

**Spark Streaming：**是 Spark 提供的对实时数据进行流式计算的组件。提供了用来操作数据流的 API，并且与 Spark Core 中的 RDD API 高度对应。

**Spark MLlib：**提供常见的机器学习\(ML\)功能的程序库。包括分类、回归、聚类、协同过滤等，还提供了模型评估、数据 导入等额外的支持功能。

**集群管理器：**Spark 设计为可以高效地在一个计算节点到数千个计算节点之间伸缩计 算。为了实现这样的要求，同时获得最大灵活性，Spark 支持在各种集群管理器\(cluster manager\)上运行，包括 Hadoop YARN、Apache Mesos，以及 Spark 自带的一个简易调度 器，叫作独立调度器。

### Spark优劣 <a id="Spark%E4%BC%98%E5%8A%A3"></a>

1、spark把运算的中间数据存放在内存，迭代计算效率更高；mapreduce的中间结果需要落地，需要保存到磁盘，这样必然会有磁盘io操做，影响性能。

2、spark容错性高，它通过弹性分布式数据集RDD来实现高效容错，RDD是一组分布式的存储在节点内存中的只读性质的数据集，这些集合是弹性的，某一部分丢失或者出错，可以通过整个数据集的计算流程的血缘关系来实现重建；mapreduce的话容错可能只能重新计算了，成本较高。

3、spark更加通用，spark提供了transformation和action这两大类的多个功能api，另外还有流式处理sparkstreaming模块、图计算GraphX等等；mapreduce只提供了map和reduce两种操作，流计算以及其他模块的支持比较缺乏。

4、spark框架和生态更为复杂，首先有RDD、血缘lineage、执行时的有向无环图DAG、stage划分等等，很多时候spark作业都需要根据不同业务场景的需要进行调优已达到性能要求；mapreduce框架及其生态相对较为简单，对性能的要求也相对较弱，但是运行较为稳定，适合长期后台运行。

大数据教程：初识Spark和Spark体系介绍 （王脸小：这篇写的非常清楚）

[https://zhuanlan.zhihu.com/p/66494957](https://zhuanlan.zhihu.com/p/66494957)

Spark面试题

[https://zhuanlan.zhihu.com/p/49169166](https://zhuanlan.zhihu.com/p/49169166)

## **Hadoop** <a id="Hadoop"></a>

[https://www.zhihu.com/question/23036370/answer/575093823](https://www.zhihu.com/question/23036370/answer/575093823)

其核心是HDFS和MapReduce，HDFS解决了文件分布式存储的问题，MapReduce解决了数据处理分布式计算的问题，HBase解决了一种数据的存储和检索。

2018年7月，Hadoop打破世界纪录，成为最快排序1TB数据的系统，用时209秒。

### **HDFS** <a id="HDFS"></a>

整个HDFS有三个重要角色：NameNode（名称节点）、DataNode（数据节点）和Client（客户机）。

### **MapReduce** <a id="MapReduce"></a>

原理讲解

[https://juejin.im/post/5bb59f87f265da0aeb7118f2](https://juejin.im/post/5bb59f87f265da0aeb7118f2)

Map：每个工作节点将 map 函数应用于本地数据，并将输出写入临时存储。主节点确保仅处理冗余输入数据的一个副本。

Shuffle：工作节点根据输出键（由 map 函数生成）重新分配数据，对数据映射排序、分组、拷贝，目的是属于一个键的所有数据都位于同一个工作节点上。

Reduce：工作节点现在并行处理每个键的每组输出数据。

**HBase**是一种构建在HDFS之上的分布式、面向列的存储系统。在需要实时读写、随机访问超大规模数据集时，可以使用Hbase

### **Hive** <a id="Hive"></a>

~~Hive就是把写的SQL语句，翻译成Mapreduce代码，然后在Hadoop上执行。~~

### **Yarn** <a id="Yarn"></a>

负责资源管理和任务调度的

1\).ResourceManager 负责所有资源的监控、分配和管理；

2\).ApplicationMaster 负责每一个具体应用程序的调度和协调；

3\).NodeManager 负责每一个节点的维护。

对于所有的applications，RM 拥有绝对的控制权和对资源的分配权。而每个AM 则会和RM协商资源，同时和NodeManager 通信来执行和监控task。

可以把yarn 理解为相当于一个分布式的操作系统平台，而mapreduce 等运算程序则相当于运行于操作系统之上的应用程序，Yarn 为这些程序提供运算所需的资源（内存、cpu）。

### 基本命令 <a id="%E5%9F%BA%E6%9C%AC%E5%91%BD%E4%BB%A4"></a>

[https://www.cnblogs.com/liu-yao/p/5-hadoop-mingling.html](https://www.cnblogs.com/liu-yao/p/5-hadoop-mingling.html)

启动Hadoop: start-all.sh

\#hdfs dfs -mkdir /input

\#hdfs dfs -ls

\#hdfs dfs -put

\#hdfs dfs -get

\#hdfs dfs -rm

\#hdfs fs -cat /input/hello.txt

文本是笔记。

原创声明，本文系作者授权云+社区发表，未经许可，不得转载。

如有侵权，请联系 yunjia\_community@tencent.com 删除。

