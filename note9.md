# 数据挖掘笔记9  

 bagging：从训练集从进行子抽样组成每个基模型所需要的子训练集，对所有基模型预测的结果进行综合产生最终的预测结果  
 
 boosting：训练过程为阶梯状，基模型按次序一一进行训练（实现上可以做到并行），基模型的训练集按照某种策略每次都进行一定的转化。对所有基模型预测的结果进行线性综合产生最终的预测结果：  
 
 stacking：将训练好的所有基模型对训练基进行预测，第j个基模型对第i个训练样本的预测值将作为新的训练集中第i个样本的第j个特征值，最后基于新的训练集进行训练。同理，预测的过程也要先经过所有基模型的预测形成新的测试集，最后再对测试集进行预测  
 

## 模型的偏差和方差
模型的偏差是一个相对来说简单的概念：训练出来的模型在训练集上的准确度。  
要解释模型的方差，首先需要重新审视模型：模型是随机变量。设样本容量为n的训练集为随机变量的集合(X1, X2, ..., Xn)，那么模型是以这些随机变量为输入的随机变量函数（其本身仍然是随机变量）：F(X1, X2, ..., Xn)。抽样的随机性带来了模型的随机性。　  
定义随机变量的值的差异是计算方差的前提条件，通常来说，我们遇到的都是数值型的随机变量，数值之间的差异再明显不过（减法运算）。但是，模型的差异性呢？我们可以理解模型的差异性为模型的结构差异，例如：线性模型中权值向量的差异，树模型中树的结构差异等。在研究模型方差的问题上，我们并不需要对方差进行定量计算，只需要知道其概念即可。　  
研究模型的方差有什么现实的意义呢？我们认为方差越大的模型越容易过拟合：假设有两个训练集A和B，经过A训练的模型Fa与经过B训练的模型Fb差异很大，这意味着Fa在类A的样本集合上有更好的性能，而Fb反之，这便是我们所说的过拟合现象。　  
我们常说集成学习框架中的基模型是弱模型，通常来说弱模型是偏差高（在训练集上准确度低）方差小（防止过拟合能力强）的模型。但是，并不是所有集成学习框架中的基模型都是弱模型。bagging和stacking中的基模型为强模型（偏差低方差高），boosting中的基模型为弱模型。　  
在bagging和boosting框架中，通过计算基模型的期望和方差，我们可以得到模型整体的期望和方差。为了简化模型，我们假设基模型的权重、方差及两两间的相关系数相等。
## 模型的独立性  

1.样本抽样：整体模型F(X1, X2, ..., Xn)中各输入随机变量（X1, X2, ..., Xn）  

2.对样本的抽样子抽样：从整体模型F(X1, X2, ..., Xn)中随机抽取若干输入随机变量成为基模型的输入随机变量　　  

假若在子抽样的过程中，两个基模型抽取的输入随机变量有一定的重合，那么这两个基模型对整体样本的抽样将不再独立，这时基模型之间便具有了相关性  

## 总结
1.使用模型的偏差和方差来描述其在训练集上的准确度和防止过拟合的能力  

2.对于bagging来说，整体模型的偏差和基模型近似，随着训练的进行，整体模型的方差降低  

3.对于boosting来说，整体模型的初始偏差较高，方差较低，随着训练的进行，整体模型的偏差降低（虽然也不幸地伴随着方差增高），当训练过度时，因方差增高，整体模型的准确度反而降低  

4.整体模型的偏差和方差与基模型的偏差和方差息息相关