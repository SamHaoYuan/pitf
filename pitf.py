import numpy as np
import pandas as pd

"""
Pairwise Interaction Tensor Factorization for Personalized Tag Recommendation
"""


class PITF:
    def __init__(self, alpha=0.0001, lamb=0.1, k=30, max_iter=100, data_shape=None, verbose=0):
        """
        数据以pandas的结构输入，为了后续的采样和顺序训练的方便，我们还是需要进行一定的处理
        :param alpha: 梯度下降速率
        :param lamb: 正则化参数
        :param k: 隐向量维度
        :param max_iter: 最大迭代次数
        :param data_shape: 训练数据维度（user，item, tag 数量）
        :param verbose:
        """
        self.alpha = alpha
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.data_shape = data_shape
        self.verbose = verbose
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
        latent_vector['u'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[0], self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[1], self.k))
        latent_vector['tu'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2], self.k))
        latent_vector['ti'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2], self.k))
        return latent_vector

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
        while r in self.trainTagSet[u][u]:
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

    def fit(self, data, validation=None):
        """
        使用BPR思想拟合模型
        :param data: 训练数据，numpy结构（3*sample)
        :param validation:  验证或测试数据，numpy结构 （3*sample)
        :return:
        """
        if self.data_shape is None:
            self.data_shape = self._calc_number_of_dimensions(data, validation)
        self.latent_vector_ = self._init_latent_vectors(self.data_shape)
        self._init_data(data, validation)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            np.random.shuffle(data)  # 打乱数据集顺序（没有必要）
            for u, i, t in data:
                nt = self._draw_negative_sample(u, i)
                y_diff = self.y(u, i, t) - self.y(u, i, nt)
                delta = 1-self._sigmoid(y_diff)
                self.latent_vector_['u'][u] += self.alpha * (delta * (self.latent_vector_['tu'][t] - self.latent_vector_['tu'][nt]) - self.lamb * self.latent_vector_['u'][u])
                self.latent_vector_['i'][i] += self.alpha * (delta * (self.latent_vector_['ti'][t] - self.latent_vector_['ti'][nt]) - self.lamb * self.latent_vector_['i'][i])
                self.latent_vector_['tu'][t] += self.alpha * (delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][t])
                self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                self.latent_vector_['ti'][t] += self.alpha * (delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][t])
                self.latent_vector_['ti'][nt] += self.alpha * (delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
            if self.verbose == 1:
                print("%s\t%s" % (self.max_iter-remained_iter, self._score(validation)))
            if remained_iter <= 0:
                break
        return self

    def y(self, i, j, k):
        return self.latent_vector_['tu'][k].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'][k].dot(self.latent_vector_['i'][j])

    def predict(self, i, j):
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][j])
        return y.argmax()

    def predict2(self, x):
        y = self.latent_vector_['u'][x[:, 0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:, 1]].dot(self.latent_vector_['ti'].T)
        return y.argmax(axis=1)
