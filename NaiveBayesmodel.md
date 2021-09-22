# 《数据挖掘》个人笔记1-徐佳诚
  
  
> @author：徐佳诚  


## 什么是数据挖掘

数据挖掘是在大型数据库中自动地发现有用信息的过程。数据挖掘技术用来探查大型数据库，发现先前未知的有用模式。数据挖掘还可以预测未来的观测结果，比如顾客在网上或实体店的消费金额。
并非所有的信息发现任务都被视为数据挖掘。例如查询任务：在数据库中查找个别记录，或查找含特定关键字的网页。这是因为这些任务可以通过与数据库管理系统或信息检索系统的简单交互来完成。而这些系统主要依赖传统的计算机科学技术，包括先进高效的索引结构和查询处理算法，有效地组织和检索大型数据存储库的信息。尽管如此，数据挖掘技术可以基于搜索结果与输入查询的相关性来提高搜索结果的质量，因此被用于提高这些系统的性能。  

数据库中的数据挖掘与知识发现数据挖掘是数据库中知识发现（Knowledge Discovery in Database，KDD）不可缺少的一部分，而KDD是将未加工的数据转换为有用信息的整个过程，由于收集和存储数据的方式多种多样，数据预处理可能是整个知识发现过程中最费力、最耗时的步骤。“结束循环”（closing the loop）通常指将数据挖掘结果集成到决策支持系统的过程。例如，在商业应用中，数据挖掘的结果所揭示的规律可以与商业活动管理工具结合，从而开展或测试有效的商品促销活动。  

这样的结合需要后处理（postprocessing）步骤，确保只将那些有效的和有用的结果集成到决策支持系统中。后处理的一个例子是可视化，它使得数据分析者可以从各种不同的视角探査数据和数据挖掘结果。在后处理阶段，还能使用统计度量或假设检验，删除虚假的数据挖掘结果。

## 数据挖掘要解决的问题

前面提到，面临大数据应用带来的挑战时，传统的数据分析技术经常遇到实际困难。下面是一些具体的问题，它们引发了人们对数据挖掘的研究。

1. 可伸缩

由于数据产生和采集技术的进步，数太字节（TB）、数拍字节（PB）甚至数艾字节（EB）的数据集越来越普遍。如果数据挖掘算法要处理这些海量数据集，则算法必须是可伸缩的。许多数据挖掘算法采用特殊的搜索策略来处理指数级的搜索问题。为实现可伸缩可能还需要实现新的数据结构，才能以有效的方式访问每个记录。

例如，当要处理的数据不能放进内存时，可能需要核外算法。使用抽样技术或开发并行和分布式算法也可以提高可伸缩程度。

2. 高维性

现在，常常会遇到具有成百上千属性的数据集，而不是几十年前常见的只具有少量属性的数据集。在生物信息学领域，微阵列技术的进步已经产生了涉及数千特征的基因表达数据。具有时间分量或空间分量的数据集也通常具有很高的维度。

例如，考虑包含不同地区的温度测量结果的数据集，如果在一个相当长的时间周期内反复地测量，则维数（特征数）的增长正比于测量的次数。为低维数据开发的传统数据分析技术通常不能很好地处理这类高维数据，如维灾难问题。此外，对于某些数据分析算法，随着维数（特征数）的增加，计算复杂度会迅速增加。


3. 异构数据和复杂数据

通常，传统的数据分析方法只处理包含相同类型属性的数据集，或者是连续的，或者是分类的。随着数据挖掘在商务、科学、医学和其他领域的作用越来越大，越来越需要能够处理异构属性的技术。

近年来，出现了更复杂的数据对象。这种非传统类型的数据如：含有文本、超链接、图像、音频和视频的Web和社交媒体数据，具有序列和三维结构的DNA数据，由地球表面不同位置、不同时间的测量值（温度、压力等）构成的气候数据。

为挖掘这种复杂对象而开发的技术应当考虑数据中的联系，如时间和空间的自相关性、图的连通性、半结构化文本和XML文档中元素之间的父子关系。

4. 数据的所有权与分布

有时，需要分析的数据不会只存储在一个站点，或归属于一个机构，而是地理上分布在属于多个机构的数据源中。这就需要开发分布式数据挖掘技术。分布式数据挖掘算法面临的主要挑战包括：

如何降低执行分布式计算所需的通信量？如何有效地统一从多个数据源获得的数据挖掘结果？如何解决数据安全和隐私问题？

5. 非传统分析

传统的统计方法基于一种假设检验模式，即提出一种假设，设计实验来收集数据，然后针对假设分析数据。但是，这一过程劳力费神。当前的数据分析任务常常需要产生和评估数千种假设，因此需要自动地产生和评估假设，这促使人们开发了一些数据挖掘技术。  


此外，数据挖掘所分析的数据集通常不是精心设计的实验的结果，并且它们通常代表数据的时机性样本（opportunistic sample），而不是随机样本（random sample）。

## 什么是特征

一个典型的机器学习任务，是通过样本的特征来预测样本所对应的值。如果样本的特征少，我们会考虑增加特征，比如Polynomial Regression就是典型的增加特征的算法。而现实中的情况往往是特征太多了，需要减少一些特征。

首先“无关特征”（irrelevant feature）。比如，通过空气的湿度，环境的温度，风力和当地人的男女比例来预测明天是否会下雨，其中男女比例就是典型的无关特征。

其实“多于特征”（redundant feature），比如，通过房屋的面积，卧室的面积，车库的面积，所在城市的消费水平，所在城市的税收水平等特征来预测房价，那么消费水平（或税收水平）就是多余特征。证据表明，税收水平和消费水平存在相关性，我们只需要其中一个特征就足够了，因为另一个能从其中一个推演出来。（若是线性相关，则用线性模型做回归时会出现多重共线性问题，将会导致过拟合）

减少特征具有重要的现实意义，不仅减少过拟合、减少特征数量（降维）、提高模型泛化能力，而且还可以使模型获得更好的解释性，增强对特征和特征值之间的理解，加快模型的训练速度，一般的，还会获得更好的性能。问题是，在面对未知领域时，很难有足够的认识去判断特征与目标之间的相关性，特征与特征之间的相关性。这时候就需要用一些数学或工程上的方法来帮助我们更好地进行特征选择。

### 常见方法
常见的方法有：

过滤法（Filter）：按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征  

包裹法（Wrapper）：根据目标函数，每次选择若干特征或者排除若干特征，直到选择出最佳的子集。  

嵌入法（Embedding）：先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。类似于Filter方法，但是是通过训练来确定特征的优劣。  




## 贝叶斯方法  

贝叶斯方法是以贝叶斯原理为基础，使用概率统计的知识对样本数据集进行分类。由于其有着坚实的数学基础，贝叶斯分类算法的误判率是很低的。贝叶斯方法的特点是结合先验概率和后验概率，即避免了只使用先验概率的主观偏见，也避免了单独使用样本信息的过拟合现象。贝叶斯分类算法在数据集较大的情况下表现出较高的准确率，同时算法本身也比较简单。  

## 朴素贝叶斯算法  

朴素贝叶斯算法（Naive Bayesian algorithm) 是应用最为广泛的分类算法之一。
朴素贝叶斯方法是在贝叶斯算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。也就是说没有哪个属性变量对于决策结果来说占有着较大的比重，也没有哪个属性变量对于决策结果占有着较小的比重。虽然这个简化方式在一定程度上降低了贝叶斯分类算法的分类效果，但是在实际的应用场景中，极大地简化了贝叶斯方法的复杂性。  

**朴素贝叶斯法对数据的要求**：数据满足特征条件独立假设，即用于分类的特征在类确定的条件下都是条件独立的。

**朴素贝叶斯法的学习过程**：基于特征条件独立假设学习输入输出的联合概率分布。即通过先验概率分布 $P(Y=c_k)$ 和条件概率分布 $P(X^{(j)}=x^{(j)}|Y=c_k)$ 实现对联合概率分布 $P(X,Y)$ 的估计。

**朴素贝叶斯法的预测过程**：基于模型，对给定的输入 x，利用贝叶斯定理求出后验概率最大的输出 y。

**朴素贝叶斯法的类别划分**：

- 用于解决分类问题的监督学习模型
- 概率模型：模型取条件概率分布形式 $P(y|x)$
- 参数化模型：假设模型参数的维度固定
- 生成模型：由数据直接学习联合概率分布 $P(X,Y)$，然后求出概条件概率分布 $P(Y|X)$

**朴素贝叶斯法的主要优点**：学习与预测的效率很高，易于实现。

**朴素贝叶斯法的主要缺点**：因为特征条件独立假设很强，分类的性能不一定很高。

> 【扩展阅读】[sklearn 中文文档：1.9 朴素贝叶斯](https://sklearn.apachecn.org/docs/master/10.html)

---

【补充说明】书中介绍的朴素贝叶斯法只适用于离散型特征，如果是连续型特征，还需要考虑连续型特征的概率密度函数。详见“延伸阅读”。

【补充说明】这里的“特征条件独立假设”即下文中“朴素贝叶斯法对条件概率分布作出的条件独立性假设”。

#### 贝叶斯定理

首先根据条件概率的定义，可得如下定理。

> **【定理 1】乘法定理 （来自浙江大学《概率论与数理统计》第四版 P. 16）**
>
> 设 $P(A)>0$，则有
>
> $$
> P(AB) = P(B|A) P(A)
> $$

接着给出划分的定义。

> **【定义 2】划分 （来自浙江大学《概率论与数理统计》第四版 P. 17）**
>
> 设 S 为试验 E 的样本空间，$B_1,B_2,\cdots,B_n$ 为 E 的一组事件。若
>
> 1. $B_i B_j = \varnothing$，$i \ne j$，$i,j=1,2,\cdots,n$；
> 2. $B_1 \cup B_2 \cup \cdots \cup B_n = S$，
>
> 则称 $B_1,B_2,\cdots,B_n$ 为样本空间 S 的一个划分。

根据划分的定义，我们知道若 $B_1,B_2,\cdots,B_n$ 为样本空间的一个划分，那么，对每次试验，事件 $B_1,B_2,\cdots,B_n$ 中必有一个且仅有一个发生。于是得到全概率公式和贝叶斯公式。

> **【定理 3】全概率公式 （来自浙江大学《概率论与数理统计》第四版 P. 18）**
>
> 设试验 E 的样本空间为 S，A 为 E 的事件，$B_1,B_2,\cdots,B_n$ 为 S 的一个划分，且 $P(B_i)>0 \ (i=1,2\cdots,n)$，则
>
> $$
> P(A) = P(A|B_1) P(B_1) + P(A|B_2) P(B_2) + \cdots + P(A|B_n) P(B_n)
> $$

> **【定理 4】贝叶斯公式 （来自浙江大学《概率论与数理统计》第四版 P. 18）**
>
> 设试验 E 的样本空间为 S，A 为 E 的事件，$B_1,B_2,\cdots,B_n$ 为 S 的一个划分，且 $P(A)>0$，$P(B_i)>0 \ (i=1,2\cdots,n)$，则
>
> $$
> P(B_i|A) = \frac{P(A|B_i) P(B_i)}{\sum_{j=1}^n P(A|B_j) P(B_j)}, \ i=1,2,\cdots,n
> $$

## 4.1.1 朴素贝叶斯法的学习与分类-基本方法

【补充说明】作出特征条件独立假设，即条件独立性的假设后，参数规模降为 $K \sum_{j=1}^n S_j$。

【补充说明】$argmax$ 函数用于计算因变量取得最大值时对应的自变量的点集。求函数 $f(x)$ 取得最大值时对应的自变量 $x$ 的点集可以写作

$$
arg \max_{x} f(x)
$$

【补充说明】在公式 4.4、4.5、4.6 中，等号右侧求和符号内的 k 并不影响求和符号外的 k，但是相同符号会带来些许歧义，或许可以将分母中的 k 改写为其他字母，例如可以将公式 4.4 改写为下式：

$$
P(Y=c_k|X=x)=\frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_i P(X=x|Y=c_i)P(Y=c_i)}
$$

## 4.1.2 朴素贝叶斯法后验概率最大化的含义

【补充说明】“期望风险”的前置知识在“1.3.2 策略”。

#### 期望的下标符号的含义

期望的下标符号似乎没有确切的定义。在[【stackexchange 的问题：Subscript notation in expectations】](https://stats.stackexchange.com/questions/72613/subscript-notation-in-expectations)中，提及使用了期望的下标符号的[【wikipedia 词条：Law of total expectation】](https://en.wikipedia.org/wiki/Law_of_total_expectation)现在也已经没有期望的下标符号了。在[@Alecos Papadopoulos](https://stats.stackexchange.com/users/28746/alecos-papadopoulos)的回答中，提到有两种可能：

第一种是将下标符号中的变量作为条件，即

$$
E_X [L(Y,f(X))] = E[L(Y,f(X))|X]
$$

第二种是将下标符号中的变量用作计算平均，即

$$
E_X [L(Y,f(X))] = \sum_{x \in X} L(Y,f(X)) P(X=x)
$$

显然，书中所使用的期望的下标符号是第二种含义。于是下式

$$
R_{exp}(f)=E_X \sum_{k=1}^K [L(c_k,f(X))]P(c_k|X)
$$

可以理解为

$$
R_{exp}(f)=E \sum_{x \in X} \bigg{[} \sum_{k=1}^K [L(c_k,f(X))]P(c_k|X=x) \bigg{]} P(X=x)
$$

#### 根据期望风险最小化准则推导后验概率最大化准则 （不使用期望的下标符号）

设 X 为 n 维向量的集合，类标记集合 Y 有 k 个值。已知期望是对联合分布 $P(X,Y)$ 取的，所以期望风险函数可以写作

$$
\begin{align}
R_{exp}(f)
& = E[L(Y,f(X))] \\
& = L(Y,f(X)) P(X,Y) \\
& = \sum_{i=1}^n \sum_{j=1}^k L(Y_j,f(X_i)) P(X=X_i,Y=Y_j) \\
& = \sum_{i=1}^n \Big{[} \sum_{j=1}^k L(Y_j,f(X_i)) P(Y=Y_j|X=X_i) \Big{]} P(X=X_i) \\
& = \sum_{i=1}^n \Big{[} \sum_{j=1}^k P(Y \ne Y_j|X=X_i) \Big{]} P(X=X_i)
\end{align}
$$

因为特征条件独立假设，所以为了使期望风险最小化，只需对 $X=x$ 逐个极小化，后续证明与书中相同，不再赘述。

## 4.2 朴素贝叶斯法的参数估计

#### 极大似然估计

极大似然估计，也称最大似然估计，其核心思想是：在已经得到试验结果的情况下，我们应该寻找使这个结果出现的可能性最大的参数值作为参数的估计值。

#### 朴素贝叶斯算法（原生 Python 实现）


```python


import collections

class NaiveBayesAlgorithmHashmap:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数
        self.n = len(x[0])  # 维度数

        count1 = collections.Counter(y)  # 先验概率的分子，条件概率的分母
        count2 = [collections.Counter() for _ in range(self.n)]  # 条件概率的分子
        for i in range(self.N):
            for j in range(self.n):
                count2[j][(x[i][j], y[i])] += 1

        # 计算先验概率和条件概率
        self.prior = {k: v / self.N for k, v in count1.items()}
        self.conditional = [{k: v / count1[k[1]] for k, v in count2[j].items()} for j in range(self.n)]

    def predict(self, x):
        best_y, best_score = 0, 0
        for y in self.prior:
            score = self.prior[y]
            for j in range(self.n):
                score *= self.conditional[j][(x[j], y)]
            if score > best_score:
                best_y, best_score = y, score
        return best_y
```

二维数组存储先验概率和条件概率。

```python

class NaiveBayesAlgorithmArray:
    """朴素贝叶斯算法（仅支持离散型数据）"""

    def __init__(self, x, y):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        table1 = [0] * self.K
        for i in range(self.N):
            table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = table1[k] / self.N

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = table2[j][k][t] / table1[k]

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= 0
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y
```


```python
>>> from code.naive_bayes import NaiveBayesAlgorithmArray
>>> from code.naive_bayes import NaiveBayesAlgorithmHashmap
>>> dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
...             (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
...             (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
...            [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
>>> naive_bayes_1 = NaiveBayesAlgorithmHashmap(*dataset)
>>> naive_bayes_1.predict([2, "S"])
-1
>>> naive_bayes_2 = NaiveBayesAlgorithmArray(*dataset)
>>> naive_bayes_2.predict([2, "S"])
-1
```

#### 贝叶斯估计（原生 Python 实现）


```python


class NaiveBayesAlgorithmWithSmoothing:
    """贝叶斯估计（仅支持离散型数据）"""

    def __init__(self, x, y, l=1):
        self.N = len(x)  # 样本数 —— 先验概率的分母
        self.n = len(x[0])  # 维度数
        self.l = l  # 贝叶斯估计的lambda参数

        # 坐标压缩（将可能存在的非数值的特征及类别转换为数值）
        self.y_list = list(set(y))
        self.y_mapping = {c: i for i, c in enumerate(self.y_list)}
        self.x_list = [list(set(x[i][j] for i in range(self.N))) for j in range(self.n)]
        self.x_mapping = [{c: i for i, c in enumerate(self.x_list[j])} for j in range(self.n)]

        # 计算可能取值数
        self.K = len(self.y_list)  # Y的可能取值数
        self.Sj = [len(self.x_list[j]) for j in range(self.n)]  # X各个特征的可能取值数

        # 计算：P(Y=ck) —— 先验概率的分子、条件概率的分母
        self.table1 = [0] * self.K
        for i in range(self.N):
            self.table1[self.y_mapping[y[i]]] += 1

        # 计算：P(Xj=ajl|Y=ck) —— 条件概率的分子
        self.table2 = [[[0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for i in range(self.N):
            for j in range(self.n):
                self.table2[j][self.y_mapping[y[i]]][self.x_mapping[j][x[i][j]]] += 1

        # 计算先验概率
        self.prior = [0.0] * self.K
        for k in range(self.K):
            self.prior[k] = (self.table1[k] + self.l) / (self.N + self.l * self.K)

        # 计算条件概率
        self.conditional = [[[0.0] * self.Sj[j] for _ in range(self.K)] for j in range(self.n)]
        for j in range(self.n):
            for k in range(self.K):
                for t in range(self.Sj[j]):
                    self.conditional[j][k][t] = (self.table2[j][k][t] + self.l) / (self.table1[k] + self.l * self.Sj[j])

    def predict(self, x):
        best_y, best_score = 0, 0
        for k in range(self.K):
            score = self.prior[k]
            for j in range(self.n):
                if x[j] in self.x_mapping[j]:
                    score *= self.conditional[j][k][self.x_mapping[j][x[j]]]
                else:
                    score *= self.l / (self.table1[k] + self.l * self.Sj[j])
            if score > best_score:
                best_y, best_score = self.y_list[k], score
        return best_y
```


```python
>>> from code.naive_bayes import NaiveBayesAlgorithmWithSmoothing
>>> dataset = [[(1, "S"), (1, "M"), (1, "M"), (1, "S"), (1, "S"),
                (2, "S"), (2, "M"), (2, "M"), (2, "L"), (2, "L"),
                (3, "L"), (3, "M"), (3, "M"), (3, "L"), (3, "L")],
               [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]]
>>> naive_bayes = NaiveBayesAlgorithmWithSmoothing(*dataset)
>>> naive_bayes.predict([2, "S"])
-1
```

## 延伸阅读

以上的朴素贝叶斯法只适用于离散型特征，如果是连续型特征，还需要考虑连续型特征的概率密度函数。当假设连续型特征服从不同的参数时，有不同的方法：

- 高斯朴素贝叶斯（GNB）：假设连续型特征服从高斯分布（正态分布），使用极大似然估计求分布参数。
- 多项分布朴素贝叶斯（MNB）：假设连续型特征服从多项式分布，使用个极大似然估计求模型参数。
- 补充朴素贝叶斯（CNB）：假设连续型特征服从多项式分布，使用来自每个类的补数的统计数据来计算模型的权重，是标准多项式朴素贝叶斯（MNB）的一种改进，特别适用于不平衡数据集。
- 伯努利朴素贝叶斯：假设连续型特征服从多重伯努利分布。

#### 支持连续型特征的朴素贝叶斯（sklearn 实现）


```python
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.naive_bayes import ComplementNB
>>> from sklearn.naive_bayes import BernoulliNB

>>> X, Y = load_breast_cancer(return_X_y=True)
>>> x1, x2, y1, y2 = train_test_split(X, Y, test_size=1 / 3, random_state=0)

>>> # 高斯朴素贝叶斯
>>> gnb = GaussianNB()
>>> gnb.fit(x1, y1)
>>> gnb.score(x2, y2)
0.9210526315789473

>>> # 多项分布朴素贝叶斯
>>> mnb = MultinomialNB()
>>> mnb.fit(x1, y1)
>>> mnb.score(x2, y2)
0.9105263157894737

>>> # 补充朴素贝叶斯
>>> mnb = ComplementNB()
>>> mnb.fit(x1, y1)
>>> mnb.score(x2, y2)
0.9052631578947369

>>> # 伯努利朴素贝叶斯
>>> bnb = BernoulliNB()
>>> bnb.fit(x1, y1)
>>> bnb.score(x2, y2)
0.6421052631578947
```
