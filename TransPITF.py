# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

"""
Pairwise Interaction Tensor Factorization for Personalized Tag Recommendation By TransTag
通过对tag隐向量进行投影，来训练PITF模型
参数为user,item,tag的隐向量，W_u和W_i分别表示tag再user,item上的投影
"""


class PITF:
    def __init__(self, alpha=0.0001, lamb=0.1, k=30, max_iter=100, init_st=0.1, data_shape=None, verbose=0):
        """
        数据以pandas的结构输入，为了后续的采样和顺序训练的方便，我们还是需要进行一定的处理
        :param alpha: 梯度下降速率
        :param lamb: 正则化参数
        :param k: 隐向量维度
        :param max_iter: 最大迭代次数
        :param init_st: 隐向量初始化标准差
        :param data_shape: 训练数据维度（user，item, tag 数量）
        :param verbose: 目前来看，用于确定是否有验证集
        """
        self.alpha = alpha
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.init_st = init_st
        self.data_shape = data_shape
        self.verbose = verbose
        self.latent_vector_ = dict()
        self.trans_matrix = dict()  # 投影矩阵
        self.trainTagSet, self.validaTagSet = dict(), dict()  # 一个用户对哪个item打过哪些tag
        self.trainUserTagSet, self.validaUserTagSet = dict(), dict()  # 一个用户使用过哪些tag
        self.trainItemTagSet, self.validaItemTagSet = dict(), dict()  # 一个Item被打过哪些tag

    def _init_latent_vectors(self, data_shape):
        """
        初始化user,item,tag隐向量
        :param data_shape: 数据维度
        :return: 隐向量初始化的结果
        """
        latent_vector = dict()
        trans_matrix = dict()
        latent_vector['u'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[0], self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[1], self.k))
        latent_vector['t'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[2], self.k))
        trans_matrix['u'] = np.random.normal(loc=0, scale=self.init_st, size=(self.k, self.k))
        trans_matrix['i'] = np.random.normal(loc=0, scale=self.init_st, size=(self.k, self.k))
        return latent_vector, trans_matrix

    def _init_data(self, data, validation=None):
        """
        遍历数据，构建user-item
        :param data:
        :return:
        """
        for u, i, t in data:
            if u not in self.trainTagSet.keys():
                self.trainTagSet[u] = dict()
            if i not in self.trainTagSet[u].keys():
                self.trainTagSet[u][i] = set()
            self.trainTagSet[u][i].add(t)
            if u not in self.trainUserTagSet.keys():
                self.trainUserTagSet[u] = set()
            if i not in self.trainItemTagSet.keys():
                self.trainItemTagSet[i] = set()
            self.trainUserTagSet[u].add(t)
            self.trainItemTagSet[i].add(t)
        if validation is not None:
            for u, i, t in validation:
                if u not in self.validaTagSet.keys():
                    self.validaTagSet[u] = dict()
                if i not in self.validaTagSet[u].keys():
                    self.validaTagSet[u][i] = set()
                self.validaTagSet[u][i].add(t)
                if u not in self.validaUserTagSet.keys():
                    self.validaUserTagSet[u] = set()
                if i not in self.validaItemTagSet.keys():
                    self.validaItemTagSet[i] = set()
                self.validaUserTagSet[u].add(t)
                self.validaItemTagSet[i].add(t)

    def _calc_number_of_dimensions(self, data, validation):
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
        if not validation is None:
            for u, i, t in validation:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if t > t_max: t_max = t
        return (u_max+1, i_max+1, t_max+1)

    def _draw_negative_sample(self, u, i):
        """
        负样本采样
        :param t: 当前正样本 index （此处应为正样本集合）
        :return: 负样本index
        这里要注意，采样应该是从当前User-item的tag集合中，找出一个没被使用过的tag
        """

        r = np.random.randint(self.data_shape[2])  # sample random index
        while r in self.trainTagSet[u][i]:
            r = np.random.randint(self.data_shape[2])  # repeat while the same index is sampled
        return r

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _score(self, data):
        """
        验证集中的准确率计算，不是top-k运算，不具备参考性
        :param data:
        :return:
        """
        if data is None: return "No validation data"
        correct = 0.
        for u, i, answer_t in data:
            predicted = self.predict(u, i)
            if predicted == answer_t: correct += 1
        return correct / data.shape[0]

    def fit(self, data, validation=None, neg_number=1):
        """
        使用BPR思想拟合模型
        :param data: 训练数据，numpy结构（3*sample)
        :param validation:  验证或测试数据，numpy结构 （3*sample)
        :param neg_number: 每个正例采样的负例次数，默认为1
        :return:
        """
        if self.data_shape is None:
            self.data_shape = self._calc_number_of_dimensions(data, validation)
        self.latent_vector_, self.trans_matrix = self._init_latent_vectors(self.data_shape)
        self._init_data(data, validation)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            print(str(self.max_iter - remained_iter))
            np.random.shuffle(data)  # 打乱数据集顺序（没有必要）
            for u, i, t in data:
                neg_sample = neg_number
                while neg_sample >= 0:
                    nt = self._draw_negative_sample(u, i)
                    neg_sample -= 1
                    y_diff = self.y(u, i, t) - self.y(u, i, nt)
                    delta = 1-self._sigmoid(y_diff)
                    self.latent_vector_['u'][u] += self.alpha * (delta * (self.latent_vector_['tu'][t] - self.latent_vector_['tu'][nt]) - self.lamb * self.latent_vector_['u'][u])
                    self.latent_vector_['i'][i] += self.alpha * (delta * (self.latent_vector_['ti'][t] - self.latent_vector_['ti'][nt]) - self.lamb * self.latent_vector_['i'][i])
                    self.latent_vector_['tu'][t] += self.alpha * (delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][t])
                    self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                    self.latent_vector_['ti'][t] += self.alpha * (delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][t])
                    self.latent_vector_['ti'][nt] += self.alpha * (delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
            if self.verbose == 1:
                self.evaluate()
                # print("%s\t%s" % (self.max_iter-remained_iter, self._score(validation)))
            if remained_iter <= 0:
                break
        return self

    def y(self, u, i, t):
        """
        隐因子向量点击
        :param u: 用户id
        :param i: item id
        :param t: 标签 id
        :return:
        """
        return self.latent_vector_['u'][u].dot(self.latent_vector_['t'][t]*self.trans_matrix['u']) + self.latent_vector_['i'][i].dot(self.latent_vector_['t'][t]*self.trans_matrix['i'])

    def predict(self, u, i):
        """
        给定用户和tag，推荐得分最高的tag
        :param u:
        :param i:
        :return: 返回 tag id
        """
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][u]) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][i])
        return y.argmax()

    def predict2(self, x):
        """
        批量输入测试数据（每一行为user-item)
        :param x: 矩阵或者张量，每一行为一个user-item测试用例
        :return
        """
        y = self.latent_vector_['u'][x[:, 0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:, 1]].dot(self.latent_vector_['ti'].T)
        return y.argmax(axis=1)

    def predict_top_k(self, u, i, k=5):
        y = (self.latent_vector_['t'] * self.trans_matrix['u']).dot(self.latent_vector_['u'][u]) + (self.latent_vector_['t']* self.trans_matrix['i']).dot(
            self.latent_vector_['i'][i])
        return y.argsort()[-k:]  # 按降序进行排列

    def evaluate(self, k=5):
        precision = 0
        recall = 0
        count = 0
        for u in self.validaTagSet.keys():
            for i in self.validaTagSet[u].keys():
                number = 0
                tags = self.validaTagSet[u][i]
                tagsNum = len(tags)
                y_pre = self.predict_top_k(u, i, k)
                for tag in y_pre:
                    if tag in tags:
                        number += 1
                    precision = precision + float(number/k)
                    recall = recall + float(number/tagsNum)
                count += 1
        precision = precision/count
        recall = recall/count
        f_score = 2 * (precision * recall) / (precision + recall)
        print("Precisions: " + str(precision))
        print("Recall: " + str(recall))
        print("F1: " + str(f_score))
        print("==================================")

