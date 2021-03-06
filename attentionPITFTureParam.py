# -*- coding: utf-8 -*-
import torch
import numpy as np
import random

random.seed(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)
from NeuralPITF import AttentionPITF, SinglePITF_Loss, DataSet
from torch.autograd import Variable
# from torch.utils import data
import torch.optim as optim
import datetime

# 在1:100正例负例采样的情况下，测试movielens数据集
ini_time = 1135429431000
movielens_all = np.genfromtxt('data/movielens/all_id_core3_train', delimiter='\t', dtype=float)
movielens_all[:, -1] = (movielens_all[:, -1] - 1135429431000) / (24 * 3600 * 1000)
movielens = movielens_all.astype(int)

movielens_test_all = np.genfromtxt('data/movielens/all_id_core3_test', delimiter='\t', dtype=float)
movielens_test_all[:, -1] = (movielens_test_all[:, -1] - 1135429431000) / (24 * 3600 * 1000)
movielens_test = movielens_test_all.astype(int)


def train(data, test):
    """
    该函数主要作用： 定义网络；定义数据，定义损失函数和优化器，计算重要指标，开始训练（训练网络，计算在测试集上的指标）
    主要需要调整的参数： m 与 gamma

    :return:
    """
    learnRate = 0.05
    lam = 0.00005
    dim = 64
    iter_ = 100
    init_st = 0.01
    m = 5
    gamma = 0.5
    batch_size = 100
    n = 1000
    # 计算numUser, numItem, numTag
    dataload = DataSet(data, test, True)
    num_user, num_item, num_tag = dataload.calc_number_of_dimensions()
    model = AttentionPITF(num_user, num_item, num_tag, dim, init_st, m, gamma).cuda()
    torch.save(model.state_dict(), 'attention_initial_params')
    # 对每个正样本进行负采样
    loss_function = SinglePITF_Loss().cuda()
    opti = optim.SGD(model.parameters(), lr=learnRate, weight_decay=lam)
    opti.zero_grad()
    # 每个epoch中的sample包含一个正样本和j个负样本
    best_result = 0
    best_result_state = model.state_dict()
    for epoch in range(iter_):
        all_data = dataload.get_sequential(num_tag, m, 100)
        all_data = all_data[:, :4 + m]
        losses = []
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for i, batch in enumerate(dataload.get_batch(all_data, batch_size)):

            # print(batch)
            # input_ = dataload.draw_negative_sample(num_tag, sample, True)
            r = model(torch.LongTensor(batch).cuda())
            opti.zero_grad()
            # print(model.embedding.userVecs.weight)
            loss = loss_function(r)
            # print(loss)
            loss.backward()
            opti.step()
            losses.append(loss.data)
            if i % n == 0:
                print("[%02d/%d] [%03d/%d] mean_loss : %0.2f" % (
                epoch, iter_, i, len(all_data) / batch_size, np.mean(losses)))
                losses = []
        precision = 0
        recall = 0
        count = 0
        validaTagSet = dataload.validaTagSet
        validaTimeList = dataload.validaUserTimeList
        for u in validaTagSet.keys():
            for i in validaTagSet[u].keys():
                number = 0
                tags = validaTagSet[u][i]
                tagsNum = len(tags)
                x_t = torch.LongTensor([u, i] + list(dataload.userShortMemory[u][:m])).cuda()
                x_t = x_t.unsqueeze(0)
                y_pre = model.predict_top_k(x_t)
                for tag in y_pre[0]:
                    if int(tag) in tags:
                        number += 1
                precision = precision + float(number / 5)
                recall = recall + float(number / tagsNum)
                count += 1
        precision = precision / count
        recall = recall / count
        if precision == 0 and recall == 0:
            f_score = 0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        print("Precisions: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f_score))
        # 将模型最好时的效果保存下来
        if f_score > best_result:
            best_result = f_score
            best_result_state = model.state_dict()
        print("best result: " + str(best_result))
        print("==================================")
    # torch.save(model, "net.pkl")
    torch.save(best_result_state, "attention_net_params.pkl")


train(movielens, movielens_test)
# model = NeuralPITF(learnRate, lam, dim, iter_, init_st, verbose=1)

# model.fit(movielens, movielens_test, 10)

# y_true = movielens_test[:, 2]
# y_pre = model.predict2(movielens_test)

# print(precision_score(y_true, y_pre))
# print(recall_score(y_true, y_pre))
# print(f1_score(y_true, y_pre))