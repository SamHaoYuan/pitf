# -*- coding: utf-8 -*-
import numpy as np
from TransPITF import TransPITF
from sklearn.metrics.classification import precision_score, recall_score, f1_score


# 在1:1 正例负例采样的情况下，测试movielens数据集

movielens_all = np.genfromtxt('data/movielens/all_id_core3_train', delimiter='\t', dtype=float)
movielens = movielens_all[:, 0:-1].astype(int)

movielens_test_all = np.genfromtxt('data/movielens/all_id_core3_test', delimiter='\t', dtype=float)
movielens_test = movielens_test_all[:, 0:-1].astype(int)

learnRate = 0.001
lam = 0.00005
lam_trans = 0.01
dim = 32
tag_dim = 64
iter_ = 100
init_st = 0.01


model = TransPITF(learnRate, lam, lam_trans, dim, tag_dim, iter_, init_st, verbose=1)
model.fit(movielens, movielens_test, 10)


# print(precision_score(y_true, y_pre))
# print(recall_score(y_true, y_pre))
# print(f1_score(y_true, y_pre))