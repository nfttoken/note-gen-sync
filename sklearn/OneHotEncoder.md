#### 1，labelencoder  LabelBinarizer ordinalencoder OneHotEncoder  pd.get_dummies有什么区别

A:LabelEncoder, OrdinalEncoder, and OneHotEncoder are all used to convert categorical data into numerical data for machine learning models, but they differ in how they handle the data and their suitability for different scenarios. LabelEncoder is for target variables, converting categories into integer labels. OrdinalEncoder is for features with inherent order, assigning numerical values based on that order. OneHotEncoder is for features without inherent order, creating binary columns for each category.

LabelEncoder、OrdinalEncoder 和 OneHotEncoder 均用于将机器学习模型的分类数据转换为数值数据，但它们在数据处理方式和适用场景方面有所不同。

从作用+接收的数据形式+加工的结果 3个方面比较：

- LabelEncoder 用于类别变量转换为整数标签，也就是对数组、DataFrame的字符abcd转化为0123，输出结果为原数组的数字化数组，如['a','b','c','d']数组经过LabelEncoder转化为[0,1,2,3]。
- LabelBinarizer 常用于y变量，是对数组、DataFrame的二值化也就是，形成独特的稀疏矩阵，如['a','b','c','d']数组经过LabelBinarizer的结果是转化为[[1 0 0 0]
  [0 1 0 0]
  [0 0 1 0]
  [0 0 0 1]]。
- OrdinalEncoder 用于具有固有顺序的特征，并根据该顺序分配数值，输入项只能是DataFrame形式的二维数组。如['a','b','c','d']数组reshape后，再经过fit和transform配对调用OrdinalEncoder()转化为。例如：

  ```pyt
  oe = OrdinalEncoder()
  oe.fit(data)
  oe1 = oe.transform(data)
  print("Oe:", oe1)
  ```
- OneHotEncoder 用于无固有顺序的特征，为每个类别创建二进制列，输入项也只能是DataFrame形式二维数组。原字符串数组需要先经过LabelEncoder再转化为带有index的二维特征矩阵，如['a','b','c','d']数组经过LabelEncoder转化为[0,1,2,3]，reshape后再调用OneHotEncoder()转化为二维的[[0,1],[1,1],[2,1],[3,1]] (后面的1，1，1，1的值表示都是独特的，前面的0123是index)
- pd.get\_dummies()是数字化后的特征，转化为特征矩阵for feature variables, coding 0 & 1 [ creating multiple dummy columns]。
-
- 结论：
- 特征  经过LabelEncoder数字化后，一般直接用pd.get\_dummies() 就可以生成送入模型的特征矩阵了，简介、方便！
- OneHotEncoder和OrdinalEncoder生成的是保持DataFrame形式的编码，不是特征矩阵。


  | 函数名                 | 作用       | 接收的数据                     | 加工的结果         |
  | ---------------------- | ---------- | ------------------------------ | ------------------ |
  | LabelBinarizer         | x,y数字化  | 不限                           | 去除重复值的数字化 |
  | LabelEncoder           | x,y数字化  | 不限                           | 保留原形式的数字化 |
  | OrdinalEncoder         | 特征数字化 | 只能二维,字符需先fit_transform | 有大小顺序         |
  | OneHotEncoder          | 特征数字化 | 只能二维,字符需先fit_transform | 无大小顺序         |
  | pd.get_dummies         | 特征数字化 | 二维                           | 可喂入模型         |
  | preprocessing.binarize | x,y二值化  | 不限                           | 根据阈值的01二值化 |

[OneHotEncoding vs LabelEncoder vs pandas get_dummies — How and Why?]([OneHotEncoding vs LabelEncoder vs pandas getdummies — How and Why? | by Harshal Soni | Medium](https://harshal-soni.medium.com/onehotencoding-vs-labelencoder-vs-pandas-get-dummies-how-and-why-b190dff7a86f))


```
