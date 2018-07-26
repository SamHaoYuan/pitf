# -*- coding: utf-8 -*-
import numpy as np
from pitf import PITF
from sklearn.metrics.classification import precision_score, recall_score, f1_score
import pandas as pd

# 在1:1 正例负例采样的情况下，测试movielens数据集

movielens = pd.read_csv('data/movielens/all_id_core3_train', sep='\t', names=['user', 'item', 'tag', 'time'])
movielens_test = pd.read_csv('data/movielens/all_id_core3_test', sep='\t', names=['user', 'item', 'tag', 'time'])
# 将movielens按照处理为序列数据

learnRate = 0.0001
lam = 0.00005
dim = 64
iter_ = 100
init_st = 0.01


model = PITF(learnRate, lam, dim, iter_, init_st, verbose=1)

model.fit(movielens, movielens_test, 100)

y_true = movielens_test[:, 2]
y_pre = model.predict2(movielens_test)

# print(precision_score(y_true, y_pre))
# print(recall_score(y_true, y_pre))
# print(f1_score(y_true, y_pre))