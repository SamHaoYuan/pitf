# -*- coding: utf-8 -*-
import numpy as np

"""
Attention-based Pairwise Interaction Tensor Factorization for Sequential Personalized Tag Recommendation 
"""


class ATPITF:
    def __init__(self, alpha=0.0001, lamb=0.1, k=30, max_iter=100, init_st=0.1, gamma=0.5, data_shape=None, verbose=0):
        """
        数据以numpy的结构输入，为了后续的采样和顺序训练的方便，我们还是需要进行一定的处理
        对于同一个用户，数据必须的用于训练，同时有两种注意力机制的方法：
        ·基于正常计算的attention,可以直接和user embedding进行权重计算，然后进行拼接
        这个方法里面，我们每次的attention，是在user embedding层面计算，user偏好
        直接用
        任意一种方法，都需要重新推导梯度下降公式
        另外，我们目前仍然是以PITF的角度进行的优化，但是需要遍历一个用户的所有序列
        后续可以考虑，从神经网络的角度进行优化

        从神经网络的角度来思考模型：
        1.Embedding layer 将每个序列的user, item, tag 映射为向量
        2.Attention Net
        3.输出层
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
        self.gamma = gamma
        self.latent_vector_ = dict()
        self.trainTagSet, self.validaTagSet = dict(), dict()  # 一个用户对哪个item打过哪些tag
        self.trainUserTagSet, self.validaUserTagSet = dict(), dict()  # 一个用户使用过哪些tag
        self.trainItemTagSet, self.validaItemTagSet = dict(), dict()  # 一个Item被打过哪些tag
        self.userTimeList, self.validaUserTimeList = dict(), dict()  # 按顺序存储每个用户的 timestamp
        self.userShortMemory = dict()  # 记录用户短期记忆

    def _init_latent_vectors(self, data_shape):
        """
        初始化user,item,tag隐向量
        :param data_shape: 数据维度
        :return: 隐向量初始化的结果
        """
        latent_vector = dict()
        latent_vector['u'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[0], self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[1], self.k))
        latent_vector['tu'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[2], self.k))
        latent_vector['ti'] = np.random.normal(loc=0, scale=self.init_st, size=(data_shape[2], self.k))
        return latent_vector

    def _init_data(self, data, validation=None):
        """
        遍历数据，构建user-item
        将user-item-tag按照时间进行排序，时间戳需要处理为天
        初始化数据序列，注意一个timestamp可能有多个tag
        :param data:
        :return:
        """
        for u, i, t, time in data:
            if u not in self.trainTagSet.keys():
                self.trainTagSet[u] = dict()
            if i not in self.trainTagSet[u].keys():
                self.trainTagSet[u][i] = set()
            self.trainTagSet[u][i].add(t)
            if u not in self.trainUserTagSet.keys():
                self.trainUserTagSet[u] = set()
            if i not in self.trainItemTagSet.keys():
                self.trainItemTagSet[i] = set()
            if u not in self.userTimeList.keys():
                self.userTimeList[u] = list()
                self.userShortMemory[u] = list()
            self.userTimeList[u].append((i, t, time))
            self.trainUserTagSet[u].add(t)
            self.trainItemTagSet[i].add(t)
        for u in self.userTimeList.keys():
            # 每一个user，处理为tag,time列表，可以根据实际情况计算权重
            self.userTimeList[u] = np.array(self.userTimeList[u])
            if len(self.userTimeList[u]) > 5:
                short_memory_list = self.userTimeList[u][:, 2].argsort()[-10:]
                for index in short_memory_list:
                    self.userShortMemory[u].append(self.userTimeList[u][index])
            else:
                self.userShortMemory[u] = self.userTimeList[u]
        if validation is not None:
            for u, i, t, time in validation:
                if u not in self.validaTagSet.keys():
                    self.validaTagSet[u] = dict()
                if i not in self.validaTagSet[u].keys():
                    self.validaTagSet[u][i] = set()
                if u not in self.validaUserTimeList.keys():
                    self.validaUserTimeList[u] = dict()
                    # 暂时不考虑多个时间戳的情况，直接进行覆盖
                    self.validaUserTimeList[u][i] = time
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
        for u, i, t, time in data:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        if not validation is None:
            for u, i, t , time in validation:
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

    def _cal_time_weight(self, pre,  now):

        return 1 + np.log10(1+np.power(10, 3)*np.exp(-0.5*(now - pre)))

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
        :param data: 训练数据，numpy结构，时间序列数据（sample_number * 4)
        :param validation:  验证或测试数据，numpy结构 （sample_number * 4)
        :param neg_number: 每个正例采样的负例次数，默认为1
        :return:
        """
        if self.data_shape is None:
            self.data_shape = self._calc_number_of_dimensions(data, validation)
        self.latent_vector_ = self._init_latent_vectors(self.data_shape)
        self._init_data(data, validation)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            print(str(self.max_iter - remained_iter))
            # np.random.shuffle(data)  # 打乱数据集顺序（没有必要）
            for u in self.userTimeList.keys():
                history = []  # 用来记录当前用户行为数
                history_tag = set()  # 记录用户已经使用过的tag
                # 根据时间先后进行遍历
                for index in self.userTimeList[u][:, 2].argsort():
                    i, t, time = self.userTimeList[u][index]
                    neg_sample = neg_number
                    c = self._cal_context(history, time)
                    while neg_sample >= 0:
                        # 负采样暂时不考虑顺序因素
                        nt = self._draw_negative_sample(u, i)
                        neg_sample -= 1
                        y_diff = self.y(u, i, t, c) - self.y(u, i, nt, c)
                        delta = 1-self._sigmoid(y_diff)
                        user_vec = self.latent_vector_['u'][u]
                        item_vec = self.latent_vector_['i'][i]
                        user_t_vec = self.latent_vector_['tu'][t]
                        user_nt_vec = self.latent_vector_['tu'][nt]
                        item_t_vec = self.latent_vector_['ti'][t]
                        item_nt_vec = self.latent_vector_['ti'][nt]
                        gra_u = self._cal_gra_t(user_vec, history)
                        self.latent_vector_['u'][u] += self.alpha * (delta *(1-self.gamma)*(user_t_vec - user_nt_vec) - self.lamb * user_vec)
                        self.latent_vector_['i'][i] += self.alpha * (delta * (item_t_vec - item_nt_vec) - self.lamb * item_vec)
                        if t not in history_tag: # 如果tag不在历史记录中，则梯度只是多了一个上下文， 否则将要考虑历史记录存在的梯度
                            self.latent_vector_['tu'][t] += self.alpha * (delta * ((1-self.gamma)*user_vec+self.gamma*c)- self.lamb * user_t_vec)
                        else:
                            gra_t = self._cal_gra_t(history, time, t)
                            self.latent_vector_['tu'][t] += self.alpha * (
                                    delta * ((1-self.gamma)*user_vec + self.gamma*(c+gra_t)) - self.lamb * user_t_vec)
                        # 负采样暂时不考虑时间因素
                        self.latent_vector_['tu'][nt] += self.alpha * (delta * -((1-self.gamma)*user_vec+self.gamma*c)- self.lamb * user_nt_vec)
                        # self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                        self.latent_vector_['ti'][t] += self.alpha * (delta * item_vec - self.lamb * item_t_vec)
                        self.latent_vector_['ti'][nt] += self.alpha * (delta * -item_vec - self.lamb * item_nt_vec)
                    if len(history) > 5:
                        history.pop(0)
                    history.append(self.userTimeList[u][index])
                    history_tag.add(t)
            if self.verbose == 1:
                self.evaluate()
                # print("%s\t%s" % (self.max_iter-remained_iter, self._score(validation)))
            if remained_iter <= 0:
                break
        return

    def _cal_softmax_weight(self, u, history):
        weights = list()
        if list(history):
            sum_w = 0
            for pre_i, pre_t, pre_time in history:
                t_vec = self.latent_vector_['tu'][pre_t]
                sum_w += np.exp(np.dot(u, t_vec))
            for pre_i, pre_t, pre_time in history:
                t_vec = self.latent_vector_['tu'][pre_t]
                weight = np.exp(np.dot(u, t_vec))/sum_w
                weights.append([pre_t, weight])
        return np.array(weights)

    def _cal_context(self, u, history):
        c = np.zeros(self.k)
        weights = self._cal_softmax_weight(u, history)
        for pre_t, weight in weights:
            c = c + weight * self.latent_vector_['tu'][pre_t]
        return c

    def _cal_gra_t(self, history, time, t):
        extra_c = np.zeros(self.k)
        for i, pre_t, pre_time in history:
            if pre_t == t:
                weight = self._cal_time_weight(pre_time, time)
                extra_c = extra_c + weight*self.latent_vector_['tu'][pre_t]
        return extra_c

    def _cal_gra_u(self, u, history):
        extra_u = np.zeros(self.k)
        weights = list()
        if list(history):
            sum_w = 0
            for pre_i, pre_t, pre_time in history:
                t_vec = self.latent_vector_['tu'][pre_t]
                sum_w += np.exp(np.dot(u, t_vec))
            for pre_i, pre_t, pre_time in history:
                t_vec = self.latent_vector_['tu'][pre_t]
                weight = np.exp(np.dot(u, t_vec)) / sum_w
                weights.append([pre_t, weight])
        return np.array(weights)
        for pre_i, pre_t, pre_time in history:
            t_vec = self.latent_vector_['tu'][pre_t]


    def y(self, u, i, t, u_m):
        """
        隐因子向量点击
        :param u: 用户id
        :param i: item id
        :param t: 标签 id
        :param time: 当前时间点
        :return:
        """
        return self.latent_vector_['tu'][t].dot((1-self.gamma)*self.latent_vector_['u'][u] + self.gamma * u_m) + self.latent_vector_['ti'][t].dot(self.latent_vector_['i'][i])

    def predict(self, u, i):
        """
        给定用户和tag，推荐得分最高的tag
        :param u:
        :param i:
        :return: 返回 tag id
        """
        # 找出该时间点前，这个用户打过的tag
        u_m = self._cal_context(u, self.userShortMemory[u])
        # 使用权重组合上下文向量
        y = self.latent_vector_['tu'].dot((1-self.gamma)*self.latent_vector_['u'][u]+self.gamma*u_m) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][i])
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
        u_m = self._cal_context(self.userShortMemory[u])
        y = self.latent_vector_['tu'].dot((1-self.gamma)*self.latent_vector_['u'][u] + self.gamma*u_m) + self.latent_vector_['ti'].dot(
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
                # 如果对一个user-item对，有多个时间戳，统一按一个处理
                time = self.validaUserTimeList[u][i]
                tagsNum = len(tags)
                y_pre = self.predict_top_k(u, i, time, k)
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

