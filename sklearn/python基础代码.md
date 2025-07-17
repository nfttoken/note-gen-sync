1，把循环所得的值存入数组，然后再从值里挑选最大最小对应的原输入值

```python
# 使用数组
candidate_max_leaf_nodes = [5,10,20,50,100,200,500]
score = [get_mae(i, train_X, val_X, train_y, val_y) for i in candidate_max_leaf_nodes]
best_mae=min(score)

best_tree_size = candidate_max_leaf_nodes[score.index(best_mae)]
# 使用字典
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
```
