Python机器学习笔记：使用sklearn做特征工程和数据挖掘

#### 完整代码及其数据，请移步小编的GitHub

　　传送门：[请点击我](https://github.com/LeBron-Jian/MachineLearningNote)

　　如果点击有误：https://github.com/LeBron-Jian/MachineLearningNote


特征处理是特征工程的核心部分，特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样式确定的步骤，更多的是工程上的经验和权衡，因此没有统一的方法，但是sklearn提供了较为完整的特征处理方法，包括数据预处理，特征选择，降维等。首次接触到sklearn，通常会被其丰富且方便的算法模型库吸引，但是这里介绍的特征处理库也非常强大！

　　经过前人的总结，特征工程已经形成了接近标准化的流程，如下图所示（此图来自[此网友](http://www.cnblogs.com/jasonfreak/p/5448385.html)，若侵权，联系我，必删除）

### ![](https://img2018.cnblogs.com/blog/1226410/201901/1226410-20190102160856049-1628723549.png)

## 1 特征来源——导入数据

　　在做数据分析的时候，特征的来源一般有两块，一块是业务已经整理好各种特征数据，我们需要去找出适合我们问题需要的特征；另一块是我们从业务特征中自己去寻找高级数据特征。

　　本文中使用sklearn中的IRIS（鸢尾花）数据集来对特征处理功能进行说明。IRIS数据集由Fisher在1936年整理，包括4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），特征值都为正浮点数，单位为厘米，目标值为鸢尾花的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））。导入IRIS数据集的代码如下：


| 12345678910 | `from` `sklearn.datasets import load_iris` `# 导入IRIS数据集``iris = load_iris()` `# 特征矩阵``data = iris.data` `# 目标向量``target = iris.target` |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |

　　从本地导入数据集的代码如下：


| 12345678 | `# 导入本地的iris数据集``dataframe = pd.read_csv(``'iris.csv'``,header=None)``iris_data = dataframe.values``# print(type(iris_data))  #<class 'numpy.ndarray'>``# 特征矩阵``data = iris_data[:,0:-1]``# 目标向量``target = iris_data[:,-1]` |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

　　

其中iris.txt的数据集如下：

[+ View Code](https://www.cnblogs.com/wj-1314/p/9600298.html#)

　　

## 

## 2，数据预处理

　对于一个项目，首先是分析项目的目的和需求，了解这个项目属于什么问题，要达到什么效果。然后提取数据，做基本的数据清洗。第三步是特征工程，这个需要耗费很大的精力。如果特征工程做得好，那么后面选择什么算法其实差异不大，反之，不管选择什么算法，效果都不会有突破性的提升。第四步是跑算法，通常情况下，我将自己会的所有的能跑的算法先跑一遍，看看效果，分析一下precesion/recall和f1-score，看看有没有什么异常（譬如有好几个算法precision特别好，但是recall特别低，这就要从数据中找原因，或者从算法中看是不是因为算法不适合这个数据），如果没有异常，那么就进行下一步，选择一两个跑的结果最好的算法进行调优。调优的方法很多，调整参数的话可以用网格搜索，随机搜索等，调整性能的话，可以根据具体的数据和场景进行具体分析，调优后再去跑一遍算法，看有没有提高，如果没有，找原因，数据还是算法问题，是数据质量不好，还是特征问题，还是算法问题？？一个一个排查，找解决方法，特征问题就回到第三步再进行特征工程，数据问题就回到第一步看数据清洗有没有遗漏，异常值是否影响了算法的结果，算法问题就回到第四步，看算法流程中哪一步出了问题。如果实在不行，可以搜一下相关的论文，看看论文中有没有解决方法。这样反复来几遍，就可以出结果了，写技术文档和分析报告，最后想产品讲解我们做的东西。然后他们再提需求，不断循环，最后代码上线，该bug。

　　直观来看，可以使用一个流程图来表示：

![](https://img2018.cnblogs.com/blog/1226410/201903/1226410-20190319150353147-1617557763.png)

#### 为什么要进行数据清洗呢？

　　我们之前实践的数据，比如iris数据集，波士顿房价数据，电影评分数据集，手写数字数据集等等，数据质量都很高，没有缺失值，没有异常点，也没有噪音。而在真实数据中，我们拿到的数据可能包含了大量的缺失值，可能包含大量的噪音，也可能因为人工录入错误导致有异常点存在，对我们挖掘出有效信息造成了一定的困扰，所以我们需要通过一些方法啊，尽量提高数据的质量。

### 2.1，分析数据

　　在实际项目中，当我们确定需求后就会去找相应的数据，拿到数据后，首先要对数据进行描述性统计分析，查看哪些数据是不合理的，也可以知道数据的基本情况。如果是销售额数据可以通过分析不同商品的销售总额，人均消费额，人均消费次数等，同一商品的不同时间的消费额，消费频次等等，了解数据的基本情况。此外可以通过作图的形式来了解数据的质量，有无异常点，有无噪音等。

　　python中包含了大量的统计命令，其中主要的统计特征函数如下图所示：

![](https://img2018.cnblogs.com/blog/1226410/201903/1226410-20190319164259377-1776574812.png)

### 2.2 处理数据（无量纲化数据的处理）

　　通过特征提取，我们能得到未经处理的特征，这时的特征可能有以下问题：

* 1，不属于同一量纲：即特征的规格不一样，不能放在一起比较。无量纲化可以解决这一问题。
* 2，信息亢余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或者“不及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和不及格。二值化可以解决这一问题。
* 3，定性特征不能直接使用：某些机器学习算法和模型只能接受定量特征的输入，那么需要将定性特征转换为定量特征。最简单的方式是为每一种定性值指定一个定量值，但是这种方式过于灵活，增加了调参的工作。通常使用哑编码的方式将定性特征转化为定量特征：假设有N种定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展特征赋值为1，其他扩展特征赋值为0，哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用哑编码后的特征可达到非线性的效果
* 4，存在缺失值：缺失值需要补充
* 5，信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似的，对于定量变量多项式化，或者进行其他的转换，都能达到非线性的效果。

　　我们使用sklearn中的preprocessing库来进行数据预处理，可以覆盖以上问题的解决方案。

　　无量纲化使不同规格的数据转换到同一规则。常见的无量纲化方法有标准化和区间缩放法。标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0,1]等。

#### 2.2.1  标准化

　　标准化需要计算特征的均值和标准差，公式表达为：

![](https://img2018.cnblogs.com/blog/1226410/201901/1226410-20190125165426099-639239470.png)

　　使用preprocessing库的StandardScaler类对数据进行标准化的代码如下：


| 1234 | `from` `sklearn.preprocessing import StandardScaler` `#标准化，返回值为标准化后的数据``StandardScaler().fit_transform(iris.data)` |
| ---- | --------------------------------------------------------------------------------------------------------------------------------- |

　　StandardScler计算训练集的平均值和标准差，以便测试数据及使用相同的变换，变换后的各维特征有0均值，单位方差（也叫z-score规范化），计算方式是将特征值减去均值，除以标准差。

　　fit  用于计算训练数据的均值和方差，后面就会使用均值和方差来转换训练数据

　　fit\_transform   不仅计算训练数据的均值和方差，还会用计算出来的均值和方差来转换训练数据，从而把数据转化成标准的正态分布。

　　transform  很显然，这只是进行转换，只是把训练数据转换成标准的正态分布。

#### 为什么要标准化？

　　通常情况下是为了消除量纲的影响。譬如一个百分制的变量与一个5分值的变量在一起怎么比较呢？只有通过数据标准化，都把他们标准到同一个标准时才具有可比性，一般标准化采用的是Z标准化，即均值为0，方差为1，当然也有其他标准化，比如0-1 标准化等等，可根据自己的数据分布情况和模型来选择。

#### 标准化适用情况

　　看模型是否具有伸缩不变性。

　　不是所有的模型都一定需要标准化，有些模型对量纲不同的数据比较敏感，譬如SVM等。当各个维度进行不均匀伸缩后，最优解与原来不等价，这样的模型，除非原始数据的分布范围本来就不叫接近，否则必须进行标准化，以免模型参数被分布范围较大或较小的数据主导。但是如果模型在各个维度进行不均匀伸缩后，最优解与原来等价，例如logistic regression等，对于这样的模型，是否标准化理论上不会改变最优解。但是，由于实际求解往往使用迭代算法，如果目标函数的形状太“扁”，迭代算法可能收敛地很慢甚至不收敛，所以对于具有伸缩不变性的模型，最好也进行数据标准化。

#### 2.2.2  区间缩放法（最小-最大规范化）

　　区间缩放法的思路有很多，常见的一种为利用两个极值进行缩放，公式表达为：

![](https://img2018.cnblogs.com/blog/1226410/201901/1226410-20190125170724019-69576460.png)

　　使用preproccessing库的MinMaxScaler类对数据进行区间缩放的代码如下：


|  |  |
| - | - |

　　区间缩放是对原始数据进行线性变换，变换到[0,1] 区间（当然也可以是其他固定最小最大值的区间）

#### 2.2.3 正则化（normalize）

　　机器学习中，如果参数过多，模型过于复杂，容易造成过拟合（overfit）。即模型在训练样本数据上表现的很好，但在实际测试样本上表现的较差，不具有良好的泛化能力，为了避免过拟合，最常用的一种方法是使用正则化，例如L1和L2正则化。

　　正则化的思想是：首先求出样本的P范数，然后该样本的所有元素都要除以该范数，这样使得最终每个样本的范数都是1，规范化（Normalization）是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，也称为归一化。

　　如下例子，将每个样本变换成unit  norm。


|  |  |
| - | - |

　　

### 2.3  对定量特征二值化

　　定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0，公式如下：

![](https://img2018.cnblogs.com/blog/1226410/201901/1226410-20190125170952765-1495568265.png)

　　使用preprocessing库的Binarizer类对数据进行二值化的代码如下：


|  |  |
| - | - |

　　给定阈值，将特征转化为0/1，最主要的是确定阈值设置。

### 2.4  对定性特征哑编码

　　由于IRIS数据集的特征皆为定量特征，故使用其目标值进行哑编码（实际上是不需要的）。使用preprocessing库的OneHotEncoder类对数据进行哑编码的代码如下：


| 1234 | `from` `sklearn.preprocessing import OneHotEncoder` `#哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据``OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))` |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

　　One-hot编码是使一种对离散特征值的编码方式，在LR模型中常用到，用于给线性模型增加非线性能力。

### 2.5  缺失值计算

　　缺失值是指粗糙数据中由于缺少信息而造成的数据的聚类，分组，删除或者截断。它指的是现有数据集中某个或者某些属性的值时不完全的。

　　缺失值在实际数据中是不可避免的问题，有的人看到缺失值就直接删除了，有的人直接赋予0值或者某一个特殊的值，那么到底该如何处理呢？对于不同的数据场景应该采取不同的策略，首先应该判断缺失值的分布情况。

　　当缺失值如果占了95%以上，可以直接去掉这个维度的数据，直接删除会对后面的算法跑的结果造成不好的影响，我们常用的方法如下：

#### 2.5.1 删除缺失值

　　如果一个样本或者变量中所包含的缺失值超过一定的比例，比如超过样本或者变量的一半，此时这个样本或者变量所含有的信息是有限的，如果我们强行对数据进行填充处理，可能会加入过大的人工信息，导致建模效果大打折扣，这种情况下，我们一般选择从数据中剔除整个样本或者变量，即删除缺失值。

#### 2.5.2 缺失值填充

* #### 随机填充法

从字面上理解就是找一个随机数，对缺失值进行填充，这种方法没有考虑任何的数据特性，填充后可能还是会出现异常值等情况。譬如将缺失值使用“Unknown”等填充，但是效果不一定好，因为算法可能会把他识别称为一个新的类别。一般情况下不建议使用。

* #### 均值填充法

寻找与缺失值变量相关性最大的那个变量把数据分成几个组，然后分别计算每个组的均值，然后把均值填入缺失的位置作为它的值，如果找不到相关性较好的变量，也可以统计变量已有数据的均值，然后把它填入缺失位置。这种方法会在一定程度上改变数据的分布。


|  |  |
| - | - |

* #### 最相似填充法

在数据集中找到一个与它最相似的样本，然后用这个样本的值对缺失值进行填充。
与均值填充法有点类似，寻找与缺失值变量（比如x）相关性最大的那个变量（比如y），然后按照变量y的值进行排序，然后得到相应的x的排序，最后用缺失值所在位置的前一个值来代替缺失值。

* #### 回归填充法（建模法）

把缺失值变量作为一个目标变量y，把缺失值变量已有部分数据作为训练集，寻找与其高度相关的变量x建立回归方程，然后把缺失值变量y所在位置对应的x作为预测集，对缺失进行预测，用预测结果来代替缺失值。

可以用回归，使用贝叶斯形式方法的基于推理的工具或者决策树归纳确定。例如利用数据集中其他数据的属性，可以构造一棵判断树，来预测缺失值的值。

* #### k近邻填充法

利用knn算法，选择缺失值的最近k个近邻点，然后根据缺失值所在的点离这几个点距离的远近进行加权平均来估计缺失值。

* #### 多重插补法

通过变量之间的关系对缺失数据进行预测，利用蒙特卡洛方法生成多个完整的数据集，在对这些数据集进行分析，最后对分析结果进行汇总处理

* #### 中位数填充法


|  |  |
| - | - |

　　

#### 2.5.3 示例——均值填充法

　　由于IRIS数据集没有缺失值，故对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失。使用preprocessing库的Imputer类对数据进行缺失值计算的代码如下：


|  |  |
| - | - |

　　

#### 2.5.4  Imputer() 处理丢失值

　　各属性必须是数值。
