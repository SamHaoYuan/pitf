import torch as t
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class InputToVector(nn.Module):
    """
    实现第一层，根据user_id, item_id 与 tag_id

    """
    def __init__(self, numUser, numItem, numTag, k, init_st):
        super(InputToVector, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k)
        self.tagItemVecs = nn.Embedding(numTag, k)
        self._init_weight(init_st)
        # self.userVecs = nn.Parameter(nn.init.normal(w_user, 0, init_std))
        # self.itemVecs = nn.Parameter(nn.init.normal(w_item, 0, init_std))
        # self.tagUserVecs = nn.Parameter(nn.init.normal(w_tag, 0, init_std))
        # self.tagItemVecs = nn.Parameter(nn.init.normal(w_tag, 0, init_std))

    def _init_weight(self, init_st):
        self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        self.tagUserVecs.weight = nn.init.normal(self.tagUserVecs.weight, 0, init_st)
        self.tagItemVecs.weight = nn.init.normal(self.tagItemVecs.weight, 0, init_st)

    def forward(self, x):
        """
        :param x: 输入数据是一个向量，包含[user,item,tag]
        :return:
        """
        # x = Variable(t.LongTensor(x)).cuda()
        user_id = x[0]
        item_id = x[1]
        tag_id = x[2]
        user_vec = self.userVecs(user_id)
        item_vec = self.itemVecs(item_id)
        tag_user_vec = self.tagUserVecs(tag_id)
        tag_item_vec = self.tagItemVecs(tag_id)
        return user_vec, item_vec, tag_user_vec, tag_item_vec


class NeuralPITF(nn.Module):
    """
    使用Pytorch，基于神经网络的思想实现PITF，
    """
    def __init__(self, numUser, numItem, numTag, k, init_st):
        super(NeuralPITF, self).__init__()
        self.embedding = InputToVector(numUser, numItem, numTag, k, init_st)

    def forward(self, x):
        user_vec, item_vec, tag_user_vec, tag_item_vec = self.embedding(x)
        r = sum(user_vec * tag_user_vec) + sum(item_vec * tag_item_vec)
        return r

    def predict_top_k(self, u, i, k=5):
        """
        给定User 和 item  根据模型返回前k个tag
        :param u:
        :param i:
        :param k:
        :return:
        """
        u_id = t.LongTensor([u])[0]
        i_id = t.LongTensor([i])[0]
        user_vec = self.embedding.userVecs(u_id)
        item_vec = self.embedding.itemVecs(i_id)
        y = user_vec.view(1, len(user_vec)).mm(self.embedding.tagUserVecs.weight.t()) + item_vec.view(1, len(item_vec)).mm(
            self.embedding.tagItemVecs.weight.t())
        return y.topk(k)[1]  # 按降序进行排列


class PITF_Loss(nn.Module):
    """
    定义PITF的loss function
    """
    def __init__(self):
        super(PITF_Loss, self).__init__()
        print("Use the BPR for Optimization")

    def forward(self, r_p, r_ne):
        return -t.log(t.sigmoid(r_p - r_ne))


class DataSet:
    """
    初始化处理数据
    :return:
    """
    def __init__(self,data, validation=None):
        self.trainTagSet, self.validaTagSet = dict(), dict()  # 一个用户对哪个item打过哪些tag
        self.trainUserTagSet, self.validaUserTagSet = dict(), dict()  # 一个用户使用过哪些tag
        self.trainItemTagSet, self.validaItemTagSet = dict(), dict()  # 一个Item被打过哪些tag
        # self.userTimeList, self.validaUserTimeList = dict(), dict()  # 按顺序存储每个用户的 timestamp
        self.data = data
        self.validation = validation
        self._init_data()

    def _init_data(self):
        """
        遍历数据，构建user-item
        将user-item-tag按照时间进行排序，时间戳需要处理为天
        初始化数据序列，注意一个timestamp可能有多个tag
        :param data:
        :return:
        """
        for u, i, t in self.data:
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
        if self.validation is not None:
            for u, i, t in self.validation:
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

    def calc_number_of_dimensions(self):
        """
        *计算数据集中，user,item,tag最大数量（数据维度，data_shape)
        :param data:
        :param validation:
        :return:
        """
        u_max = -1
        i_max = -1
        t_max = -1
        for u, i, t in self.data:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        for u, i, t in self.validation:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        return u_max + 1, i_max + 1, t_max + 1

    def draw_negative_sample(self, num_tag, pos):
        """
        负样本采样
        :param t: 当前正样本 index （此处应为正样本集合）
        :return: 负样本index
        这里要注意，采样应该是从当前User-item的tag集合中，找出一个没被使用过的tag
        """

        u, i = pos[0], pos[1]
        r = np.random.randint(num_tag)  # sample random index
        while r in self.trainTagSet[u][i]:
            r = np.random.randint(num_tag)  # repeat while the same index is sampled
        return [u, i, r]