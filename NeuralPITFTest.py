# -*- coding: utf-8 -*-
import numpy as np
from NeuralPITF import NeuralPITF, PITF_Loss
import torch
from torch.utils import data
import torch.optim as optim

# 在1:1 正例负例采样的情况下，测试movielens数据集

movielens_all = np.genfromtxt('data/movielens/all_id_core3_train', delimiter='\t', dtype=float)
movielens = movielens_all[:, 0:-1].astype(int)

movielens_test_all = np.genfromtxt('data/movielens/all_id_core3_test', delimiter='\t', dtype=float)
movielens_test = movielens_test_all[:, 0:-1].astype(int)

dataloader = data.DataLoader()

def _calc_number_of_dimensions(data, validation):
    """
    *计算数据集中，user,item,tag最大数量（数据维度，data_shape)
    :param data:
    :param validation:
    :return:
    """
    u_max = -1
    i_max = -1
    t_max = -1
    for u, i, t in data:
        if u > u_max: u_max = u
        if i > i_max: i_max = i
        if t > t_max: t_max = t
    for u, i, t in validation:
        if u > u_max: u_max = u
        if i > i_max: i_max = i
        if t > t_max: t_max = t
    return u_max + 1, i_max + 1, t_max + 1


def train(data, test):
    """
    该函数主要作用： 定义网络；定义数据，定义损失函数和优化器，计算重要指标，开始训练（训练网络，计算在测试集上的指标）
    :return:
    """
    learnRate = 0.01
    lam = 0.00005
    dim = 64
    iter_ = 500
    init_st = 0.01
    # 计算numUser, numItem, numTag
    numUser, numItem, numTag = _calc_number_of_dimensions(data, test)
    model = NeuralPITF(numUser, numItem,numTag, dim, init_st).cuda()
    # 对每个正样本进行负采样
    loss_function = PITF_Loss()
    opti = optim.SGD(model.parameters(), lr=learnRate, weight_decay=lam)
    # 每个epoch中的sample包含一个正样本和j个负样本
    for epoch in range(iter_):
        for sample in data:
            ne_sample = 1





# model = NeuralPITF(learnRate, lam, dim, iter_, init_st, verbose=1)

# model.fit(movielens, movielens_test, 10)

# y_true = movielens_test[:, 2]
# y_pre = model.predict2(movielens_test)

# print(precision_score(y_true, y_pre))
# print(recall_score(y_true, y_pre))
# print(f1_score(y_true, y_pre))