# -*- coding: utf-8 -*-
import numpy as np
from NeuralPITF import NeuralPITF, PITF_Loss, DataSet
from torch.autograd import Variable
# from torch.utils import data
import torch.optim as optim
import torch

# 在1:1 正例负例采样的情况下，测试movielens数据集

movielens_all = np.genfromtxt('data/movielens/all_id_core3_train', delimiter='\t', dtype=float)
movielens = movielens_all[:, 0:-1].astype(int)

movielens_test_all = np.genfromtxt('data/movielens/all_id_core3_test', delimiter='\t', dtype=float)
movielens_test = movielens_test_all[:, 0:-1].astype(int)


def train(data, test):
    """
    该函数主要作用： 定义网络；定义数据，定义损失函数和优化器，计算重要指标，开始训练（训练网络，计算在测试集上的指标）
    :return:
    """
    learnRate = 0.01
    lam = 0.00005
    dim = 64
    iter_ = 100
    init_st = 0.01
    # 计算numUser, numItem, numTag
    dataload = DataSet(data, test)
    num_user, num_item, num_tag = dataload.calc_number_of_dimensions()
    model = NeuralPITF(num_user, num_item, num_tag, dim, init_st).cuda()
    # 对每个正样本进行负采样
    loss_function = PITF_Loss().cuda()
    opti = optim.SGD(model.parameters(), lr=learnRate, weight_decay=lam)
    opti.zero_grad()
    # 每个epoch中的sample包含一个正样本和j个负样本
    for epoch in range(iter_):
        losses=[]
        n = 0
        for sample in data:
            n += 1
            numNeg = 10
            input_ = sample
            while numNeg > 0:
                numNeg -= 1
                neg = dataload.draw_negative_sample(num_tag, input_)
                sample = Variable(torch.LongTensor(input_)).cuda()
                neg = Variable(torch.LongTensor(neg)).cuda()
                r_p = model(sample)
                r_n = model(neg)
                opti.zero_grad()
                # print(model.embedding.userVecs.weight)
                loss = loss_function(r_p, r_n)
                loss.backward()
                opti.step()
                losses.append(loss.data)
            if n % 1000 == 0:
                print ("the loss of %s sample in %s iter is : " %(str(epoch), str(n)) + str(np.mean(losses)))
        precision = 0
        recall = 0
        count = 0
        validaTagSet = dataload.validaTagSet
        for u in validaTagSet.keys():
            for i in validaTagSet[u].keys():
                number = 0
                tags = validaTagSet[u][i]
                tagsNum = len(tags)
                y_pre = model.predict_top_k(u, i, 5)
                for tag in y_pre:
                    if tag in tags:
                        number += 1
                precision = precision + float(number / 5)
                recall = recall + float(number / tagsNum)
                count += 1
        precision = precision / count
        recall = recall / count
        if precision==0 and recall == 0:
            f_score = 0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        print("Precisions: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f_score))
        print("==================================")


train(movielens, movielens_test)
# model = NeuralPITF(learnRate, lam, dim, iter_, init_st, verbose=1)

# model.fit(movielens, movielens_test, 10)

# y_true = movielens_test[:, 2]
# y_pre = model.predict2(movielens_test)

# print(precision_score(y_true, y_pre))
# print(recall_score(y_true, y_pre))
# print(f1_score(y_true, y_pre))