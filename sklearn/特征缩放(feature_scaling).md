2 特征缩放(feature scaling)

- MinMaxScaler   基于最大最小值，使用z-score将数据转换到0,1区间上的，常见用于**神经网络**。

  ```python
  X = np.array( [0, 0.5, 1, 1.5, 2, 100] )
  X_scale = MinMaxScaler().fit_transform( X.reshape(-1,1) )
  # 结果：
  array([[0.   ],
         [0.005],
         [0.01 ],
         [0.015],
         [0.02 ],
         [1.   ]])
  ```
- StandardScaler  基于特征矩阵的列，使用z-score将属性值转换至服从正态分布的同一量纲下,常用与基于**正态分布**的算法，比如回归。

  ```python
  X_scale = StandardScaler().fit_transform( X.reshape(-1,1) )
  array([[-0.47424487],
         [-0.46069502],
         [-0.44714517],
         [-0.43359531],
         [-0.42004546],
         [ 2.23572584]])
  ```

* Normalizer （基于矩阵的行，将样本向量转换为单位向量）
  其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准
  常见用于**文本分类**和**聚类**、logistic回归中也会使用，有效防止过拟合
