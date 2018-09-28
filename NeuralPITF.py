# -*- coding: utf-8 -*-
import torch as t
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import random
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
        u_id = t.LongTensor([u]).cuda()[0]
        i_id = t.LongTensor([i]).cuda()[0]
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
    区分是否带有时间戳的情况
    :return:
    """
    def __init__(self, data, validation=None, is_timestamp=False):
        self.trainTagSet, self.validaTagSet = dict(), dict()  # 一个用户对哪个item打过哪些tag
        self.trainUserTagSet, self.validaUserTagSet = dict(), dict()  # 一个用户使用过哪些tag
        self.trainItemTagSet, self.validaItemTagSet = dict(), dict()  # 一个Item被打过哪些tag
        self.userTimeList, self.validaUserTimeList = dict(), dict()  # 按顺序存储每个用户的 timestamp
        self.userShortMemory = dict()  # 记录预测用户历史序列
        self.data = data
        self.validation = validation
        self.is_timestamp = is_timestamp
        self._init_data() if not is_timestamp else self._init_timestamp_data()

    def _init_data(self):
        """
        遍历数据，构建user-item
        :param data:
        :return:
        """
        for u, i, tag in self.data:
            if u not in self.trainTagSet.keys():
                self.trainTagSet[u] = dict()
            if i not in self.trainTagSet[u].keys():
                self.trainTagSet[u][i] = set()
            self.trainTagSet[u][i].add(tag)
            if u not in self.trainUserTagSet.keys():
                self.trainUserTagSet[u] = set()
            if i not in self.trainItemTagSet.keys():
                self.trainItemTagSet[i] = set()
            self.trainUserTagSet[u].add(tag)
            self.trainItemTagSet[i].add(tag)
        if self.validation is not None:
            for u, i, tag in self.validation:
                if u not in self.validaTagSet.keys():
                    self.validaTagSet[u] = dict()
                if i not in self.validaTagSet[u].keys():
                    self.validaTagSet[u][i] = set()
                self.validaTagSet[u][i].add(tag)
                if u not in self.validaUserTagSet.keys():
                    self.validaUserTagSet[u] = set()
                if i not in self.validaItemTagSet.keys():
                    self.validaItemTagSet[i] = set()
                self.validaUserTagSet[u].add(tag)
                self.validaItemTagSet[i].add(tag)

    def _init_timestamp_data(self):
        """
        遍历数据
        将user-item-tag按照时间进行排序，时间戳需要处理为天
        初始化数据序列，注意一个timestamp可能有多个tag
        因为对于序列，不足长度的要采取补0的措施，因此所有tag_id需要+1
        :return:
        """
        for u, i, tag, time in self.data:
            tag = tag+1
            if u not in self.trainTagSet.keys():
                self.trainTagSet[u] = dict()
            if i not in self.trainTagSet[u].keys():
                self.trainTagSet[u][i] = set()
            self.trainTagSet[u][i].add(tag)
            if u not in self.trainUserTagSet.keys():
                self.trainUserTagSet[u] = set()
            if i not in self.trainItemTagSet.keys():
                self.trainItemTagSet[i] = set()
            if u not in self.userTimeList.keys():
                self.userTimeList[u] = list()
                # self.userShortMemory[u] = np.zeros(m)
            self.userTimeList[u].append((i, tag, time))
            self.trainUserTagSet[u].add(tag)
            self.trainItemTagSet[i].add(tag)
        if self.validation is not None:
            for u, i, tag, time in self.validation:
                tag = tag+1
                if u not in self.validaTagSet.keys():
                    self.validaTagSet[u] = dict()
                if i not in self.validaTagSet[u].keys():
                    self.validaTagSet[u][i] = set()
                if u not in self.validaUserTimeList.keys():
                    self.validaUserTimeList[u] = dict()
                    # 暂时不考虑多个时间戳的情况，直接进行覆盖
                    self.validaUserTimeList[u][i] = time
                self.validaTagSet[u][i].add(tag)
                if u not in self.validaUserTagSet.keys():
                    self.validaUserTagSet[u] = set()
                if i not in self.validaItemTagSet.keys():
                    self.validaItemTagSet[i] = set()
                self.validaUserTagSet[u].add(tag)
                self.validaItemTagSet[i].add(tag)

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
        if not self.is_timestamp:
            for u, i, tag in self.data:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if tag > t_max: t_max = tag
            for u, i, tag in self.validation:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if tag > t_max: t_max = tag
            return u_max + 1, i_max + 1, t_max + 2
        else:
            for u, i, tag, time in self.data:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if tag > t_max: t_max = tag
            for u, i, tag ,time in self.validation:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if tag > t_max: t_max = tag
            return u_max + 1, i_max + 1, t_max + 2

    def get_batch(self, all_data, batch_size):
        random.shuffle(all_data)
        sindex = 0
        eindex = batch_size
        while eindex < len(all_data):
            batch = all_data[sindex: eindex]
            # 对每一行数据进行负采样，组成新的sample
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch

        if eindex >= len(all_data):
            batch = all_data[sindex:]
            yield batch

    def draw_negative_sample(self, num_tag, pos, pairwise=False):
        """
        负样本采样
        这里要注意，采样应该是从当前User-item的tag集合中，找出一个没被使用过的tag
        """

        u, i, tag = pos[0], pos[1], pos[2]
        r = np.random.randint(num_tag)  # sample random index
        while r in self.trainTagSet[u][i]:
            r = np.random.randint(num_tag)  # repeat while the same index is sampled
        return [u, i, tag, r] if pairwise else [u, i, r]

    def get_negative_samples(self, num_tag, pairwise=False, num=10):
        all_data = []
        for pos in self.data:
            k = num
            while k > 0:
                k -=1
                one_sample = self.draw_negative_sample(num_tag, pos, pairwise)
                all_data.append(one_sample)
        return np.array(all_data)

    def get_sequential(self, num_tag, m=5, num=10):
        """
        将输入数据中的每个user，按照时间顺序进行排序以后，返回m长度的记忆
        原始数据中，必须带有时间戳
        :param num_tag: 为负采样准备的参数，tag数量
        :param m: 记忆序列长度
        ：:param num: 每个整理的负采样数量
        :return: [u, i, t, neg_t, [m_1,m_2,m_3,m_4,m_5.....],timestamp, [t_1,t_2,t_3,t_4,t_5.....]]
        """
        seq_data = []
        for u in self.userTimeList.keys():
            # 每一个user，处理为tag,time列表，可以根据实际情况计算权重
            user_seqs = np.array(self.userTimeList[u]) # num * 3
            order = user_seqs[:, 2].argsort()
            user_seqs = user_seqs[order, :]
            for i in range(len(user_seqs)):
                # i 是当前时刻的训练数据
                item = user_seqs[i][0]
                tag = user_seqs[i][1]
                timestamp = user_seqs[i][2]
                tag_memory = np.zeros(m)
                timestamp_memory = np.zeros(m)
                if i < m:
                    tag_memory[m-i:] = user_seqs[:i, 1]
                    timestamp_memory[m-i:] = user_seqs[:i, 2]
                else:
                    tag_memory = user_seqs[i-m:i, 1]
                    timestamp_memory = user_seqs[i-m:i, 2]
                j = 0
                while j < num:
                    j += 1
                    pairwise_sample = self.draw_negative_sample(num_tag, [u, item, tag], True)
                    # seq_data.append([u, item, tag, neg_t, timestamp, tag_memory, timestamp_memory])
                    seq_data.append(pairwise_sample+list(tag_memory) + [timestamp] + list(timestamp_memory))
            self.userShortMemory[u] = np.zeros(2*m)
            if len(self.userTimeList[u]) > m:
                self.userShortMemory[u][0:m] = user_seqs[-m:, 1]
                self.userShortMemory[u][m:] = user_seqs[-m:, 2]
                # short_memory_list = user_seqs[-m:]
            else:
                length = len(self.userTimeList[u])
                self.userShortMemory[u][m - length:m] = user_seqs[:, 1]
                self.userShortMemory[u][2*m - length:] = user_seqs[:, 2]
        return np.array(seq_data)


class SinglePITF(nn.Module):
    """
    使用Pytorch，基于神经网络的思想实现PITF，注意，输入数据为一个pairwise(u,i,pos_t, neg_t)
    """
    def __init__(self, numUser, numItem, numTag, k, init_st):
        super(SinglePITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k)
        self.tagItemVecs = nn.Embedding(numTag, k)
        self._init_weight(init_st)

    def _init_weight(self, init_st):
        self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        self.tagUserVecs.weight = nn.init.normal(self.tagUserVecs.weight, 0, init_st)
        self.tagItemVecs.weight = nn.init.normal(self.tagItemVecs.weight, 0, init_st)

    def forward(self, x):
        """
        user_id = x[0]
        item_id = x[1]
        pos_id = x[2]
        neg_id = x[3]

        user_vec = self.userVecs(user_id)
        item_vec = self.itemVecs(item_id)
        tag_user_vec = self.tagUserVecs(pos_id)
        tag_item_vec = self.tagItemVecs(pos_id)
        neg_tag_user_vec = self.tagUserVecs(neg_id)
        neg_tag_item_vec = self.tagItemVecs(neg_id)
        r = t.sum(user_vec * tag_user_vec) + t.sum(item_vec * tag_item_vec) - (t.sum(user_vec * neg_tag_user_vec) + t.sum(
            item_vec*neg_tag_item_vec
        ))
        return r
        """
        if len(x.size()) == 1:
            x = x.view(1, len(x))
        user_id = x[:, 0]
        item_id = x[:, 1]
        pos_id = x[:, 2]
        neg_id = x[:, 3]

        user_vec = self.userVecs(user_id)
        item_vec = self.itemVecs(item_id)
        tag_user_vec = self.tagUserVecs(pos_id)
        tag_item_vec = self.tagItemVecs(pos_id)
        neg_tag_user_vec = self.tagUserVecs(neg_id)
        neg_tag_item_vec = self.tagItemVecs(neg_id)
        r = t.sum(user_vec * tag_user_vec, dim=1) + t.sum(item_vec * tag_item_vec, dim=1) - (
                t.sum(user_vec * neg_tag_user_vec, dim=1) + t.sum(item_vec*neg_tag_item_vec, dim=1))
        return r

    def predict_top_k(self, u, i, k=5):
        """
        给定User 和 item  根据模型返回前k个tag
        :param u:
        :param i:
        :param k:
        :return:
        """
        u_id = t.LongTensor([u]).cuda()[0]
        i_id = t.LongTensor([i]).cuda()[0]
        user_vec = self.userVecs(u_id)
        item_vec = self.itemVecs(i_id)
        y = user_vec.view(1, len(user_vec)).mm(self.tagUserVecs.weight.t()) + item_vec.view(1, len(item_vec)).mm(
            self.tagItemVecs.weight.t())
        return y.topk(k)[1]  # 按降序进行排列


class SinglePITF_Loss(nn.Module):
    """
    定义PITF的loss function
    """
    def __init__(self):
        super(SinglePITF_Loss, self).__init__()
        print("Use the BPR for Optimization")

    def forward(self, r):
        return t.sum(-t.log(t.sigmoid(r)))


class TransPITF(nn.Module):
    """
    不同于PITF，tag只有一个对应的embedding，然后分别在user与item层面上进行投影
    """
    def __init__(self, numUser, numItem, numTag, k, init_st):
        super(TransPITF, self).__init__()
        self.k = k
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagVecs = nn.Embedding(numTag, k)
        self.userTransM = nn.Linear(k, k)
        self.itemTransM = nn.Linear(k, k)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _init_weight(self, init_st):
        self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        self.tagVecs.weight = nn.init.normal(self.tagUserVecs.weight, 0, init_st)
        self.userTransM.weight = nn.init.uniform_(self.userTransM, -np.sqrt(3/self.k), np.sqrt(3/self.k))
        self.itemTransM.weight = nn.init.uniform_(self.itemTransM, -np.sqrt(3/self.k), np.sqrt(3/self.k))

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.view(1, len(x))
        user_id = x[:, 0]
        item_id = x[:, 1]
        pos_id = x[:, 2]
        neg_id = x[:, 3]

        user_vec = self.userVecs(user_id)
        item_vec = self.itemVecs(item_id)
        tag_vec = self.tagVecs(pos_id)
        neg_tag_vec = self.tagVecs(neg_id)

        user_tag_vec = self.sigmoid(self.userTransM(tag_vec))
        item_tag_vec = self.sigmoid(self.itemTransM(tag_vec))
        neg_user_tag_vec = self.sigmoid(self.userTransM(neg_tag_vec))
        neg_item_tag_vec = self.sigmoid(self.itemTransM(neg_tag_vec))

        r = t.sum(user_vec * user_tag_vec, dim=1) + t.sum(item_vec * item_tag_vec, dim=1) - (
                t.sum(user_vec * neg_user_tag_vec, dim=1) + t.sum(item_vec * neg_item_tag_vec, dim=1))
        return r

    def predict_top_k(self, u, i, k=5):
        """
        给定User 和 item  根据模型返回前k个tag
        :param u:
        :param i:
        :param k:
        :return:
        """
        u_id = t.LongTensor([u]).cuda()[0]
        i_id = t.LongTensor([i]).cuda()[0]
        user_vec = self.userVecs(u_id)
        item_vec = self.itemVecs(i_id)
        y = user_vec.view(1, len(user_vec)).mm(self.sigmoid(self.userTransM(self.tagVecs.weight)).t()) + item_vec.view(1, len(
            item_vec)).mm(self.sigmoid(self.itemTransM(self.tagVecs.weight)).t())
        return y.topk(k)[1]  # 按降序进行排列


class TimeAttentionPITF(nn.Module):
    """
    基于时间衰减的attention权重的PITF
    这个网络中，我们以序列的形式进行输入
    1.将用户的行为序列，分割为固定长度大小，如一条sample，包含一个用户的 m 次 tag 行为（需要注意，如果不足m次，需要进行
    让网络忽略
    2.每一条sample的每一次行为进行embedding
    3.根据时间长度计算权重，并分配给sample中的tag，然后和user建立组合向量作为user_vec
    4.其他内容不变

    出于加快训练的目的，应当思考如何进行向量化运算
    """
    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma):
        super(TimeAttentionPITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.tagItemVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.m = m
        self.k = k
        self.gamma = gamma
        self._init_weight(init_st)

    def _init_weight(self, init_st):
        self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)

    def forward(self, x):
        """
        一条sample为 [user, item, tag, ne_tag, m_1,m_2,m_3,...,t_1,t_2,t_3...]
        :param x: m 个 [user, item, tag] 组成的 sample
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_vec_ids = x[:, 2]
        neg_tag_vec_ids = x[:, 3]
        # timestamp = x[:, 4]
        tag_memory_ids = x[:, 4:4+self.m]
        timestamp = x[:, 4+self.m]
        time_memory = x[:, -self.m:]
        user_vecs = self.userVecs(user_vec_ids)
        item_vecs = self.itemVecs(item_vec_ids)
        user_tag_vecs = self.tagUserVecs(tag_vec_ids)
        item_tag_vecs = self.tagItemVecs(tag_vec_ids)
        neg_tag_user_vec = self.tagUserVecs(neg_tag_vec_ids)
        neg_tag_item_vec = self.tagItemVecs(neg_tag_vec_ids)
        tag_memory_vecs = self.tagUserVecs(tag_memory_ids)

        h = self.TimeAttention(tag_memory_vecs, time_memory, timestamp)
        mix_user_vecs = (1-self.gamma) * user_vecs + self.gamma * h
        r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
                t.sum(mix_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs*neg_tag_item_vec, dim=1))
        return r

    def TimeAttention(self, history_vecs, timestamps, now_time):
        """

        :param history_vecs: Tensor, (batch_size, m, 64)
        :param timestamps: Tensor (batch_size, m) 记录每个tag的时间
        :return: c 历史行为组合的向量(batch_size, 64)
        """
        # batch_size = len(history_vecs)
        # c = np.zeros(self.k)
        weight = self._cal_weight(timestamps, now_time)
        weight = weight.unsqueeze(1)
        c = t.bmm(weight, history_vecs)
        return c.squeeze(1)

    def _cal_weight(self, history_times, now_time):
        """

        :param history_times:
        :param now_time:
        :return: tensor, (batch_size, m)
        """
        # batch_size = len(now_time)
        a = t.exp((-0.5*(now_time.unsqueeze(1) - history_times)).type(t.FloatTensor)).cuda()
        # a = a.type(t.LongTensor).cuda()
        sum_weight = t.sum(a, dim=1)
        return a/sum_weight.view(len(sum_weight), 1)

    def predict_top_k(self, x, k=5):
        """
        给定User 和 item  记忆序列，根据模型返回前k个tag
        输入sample:[u,i,m_1,m_2....m_j, t_1,t_2,t_3,...., t]
        :param u:
        :param i:
        :param h_l
        :param k:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        # timestamp = x[:, 2]
        tag_memory_ids = x[:, 2:2 + self.m]
        timestamp = x[:, -1]
        time_memory = x[:, -self.m-1:-1]

        user_vec = self.userVecs(user_vec_ids)
        item_vec = self.itemVecs(item_vec_ids)
        h_vecs = self.tagUserVecs(tag_memory_ids)
        h = self.TimeAttention(h_vecs, time_memory, timestamp)
        mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        y = mix_user_vec.mm(self.tagUserVecs.weight.t()) + item_vec.mm(self.tagItemVecs.weight.t())
        return y.topk(k)[1]  # 按降序进行排列


class AttentionPITF(nn.Module):
    """
    直接基于attention机制的PITF
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """
    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma):
        super(AttentionPITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.tagItemVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.attentionMLP = nn.Linear(k, k)
        self.relu = nn.ReLU()
        self.m = m
        self.k = k
        self.gamma = gamma
        self._init_weight(init_st)
        
    def _init_weight(self, init_st):
        self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        # self.tagUserVecs.weight = nn.init.normal(self.tagUserVecs.weight, 0, init_st)
        # self.tagItemVecs.weight = nn.init.normal(self.tagItemVecs.weight, 0, init_st)

    def forward(self, x):
        """
        user与attention之后的h组合为新的user embedding或：
        直接将attention之后的向量作为user embedding
        :param x:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_vec_ids = x[:, 2]
        neg_tag_vec_ids = x[:, 3]
        history_ids = x[:, -self.m:]

        user_vecs = self.userVecs(user_vec_ids)
        item_vecs = self.itemVecs(item_vec_ids)
        user_tag_vecs = self.tagUserVecs(tag_vec_ids)
        item_tag_vecs = self.tagItemVecs(tag_vec_ids)
        neg_tag_user_vec = self.tagUserVecs(neg_tag_vec_ids)
        neg_tag_item_vec = self.tagItemVecs(neg_tag_vec_ids)
        tag_history_vecs = self.tagUserVecs(history_ids)

        h = self.attention(user_vecs, tag_history_vecs)  # batch * k
        mix_user_vecs = (1-self.gamma) * user_vecs + self.gamma * h
        mix_user_vecs = mix_user_vecs.unsqueeze(1)
        user_tag_vecs = user_tag_vecs.unsqueeze(2)
        item_vecs = item_vecs.unsqueeze(1)
        item_tag_vecs = item_tag_vecs.unsqueeze(2)
        neg_tag_user_vec = neg_tag_user_vec.unsqueeze(2)
        neg_tag_item_vec = neg_tag_item_vec.unsqueeze(2)
        r = t.bmm(mix_user_vecs, user_tag_vecs) + t.bmm(item_vecs, item_tag_vecs) - (
                t.bmm(mix_user_vecs, neg_tag_user_vec) + t.bmm(item_vecs, neg_tag_item_vec))
        # r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
        #        t.sum(mix_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
        return r

    def attention(self, u_vec, h_vecs):
        """
        这一层，我们可以尝试多种attention的方法：
        方案1：query 为user， key为历史tag, value同样为历史tag，然后将user+tag组成为新的user embedding
        方案2：query为user, key和value为tag进行一层MLP后的结果（从TransPITF的结果来看，进行MLP和激活后，或许可以得到一个好点的结果）
        :param u_vec: （batch_size, k)
        :param h_vecs (batch_size, m, k)
        :return:
        """
        # batch_size = u_vec.size()[0]
        # h_u_vec_ = self.relu(self.attentionMLP(u_vec))
        # u_vec_ = h_u_vec_.unsqueeze(2)
        u_vec_ = u_vec.unsqueeze(2)
        tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
        alpha = nn.functional.softmax(t.bmm(tag_h_vecs, u_vec_).squeeze(2), 1)
        alpha = alpha.unsqueeze(1)
        h = t.bmm(alpha, h_vecs)
        return h.squeeze(1)

    def predict_top_k(self, x, k=5):
        """
        给定User 和 item  记忆序列，根据模型返回前k个tag
        输入sample:[u,i,m_1,m_2....m_j]
        :param u:
        :param i:
        :param k:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_memory_ids = x[:, -self.m:]

        user_vec = self.userVecs(user_vec_ids)
        item_vec = self.itemVecs(item_vec_ids)
        h_vecs = self.tagUserVecs(tag_memory_ids)
        h = self.attention(user_vec, h_vecs)
        mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        y = mix_user_vec.mm(self.tagUserVecs.weight.t()) + item_vec.mm(self.tagItemVecs.weight.t())

        return y.topk(k)[1]  # 按降序进行排列


class RNNAttentionPITF(AttentionPITF):
    """
    基于RNN+attention机制的标签推荐模型
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """

    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma):
        super(RNNAttentionPITF, self).__init__(numUser, numItem, numTag, k, init_st, m, gamma)
        self.lstm = nn.LSTM(k, k, batch_first=True, dropout=0.5)
        self.gru = nn.GRU(k,k,batch_first=True, dropout=0.5)

    def forward(self, x):
        """
        user与attention之后的h组合为新的user embedding或：
        直接将attention之后的向量作为user embedding
        :param x:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_vec_ids = x[:, 2]
        neg_tag_vec_ids = x[:, 3]
        history_ids = x[:, -self.m:]

        user_vecs = self.userVecs(user_vec_ids)
        item_vecs = self.itemVecs(item_vec_ids)
        user_tag_vecs = self.tagUserVecs(tag_vec_ids)
        item_tag_vecs = self.tagItemVecs(tag_vec_ids)
        neg_tag_user_vec = self.tagUserVecs(neg_tag_vec_ids)
        neg_tag_item_vec = self.tagItemVecs(neg_tag_vec_ids)
        tag_history_vecs = self.tagUserVecs(history_ids)

        # out, out_final = self.gru(tag_history_vecs)
        out, out_final = self.gru(tag_history_vecs)
        
        h = self.attention(user_vecs, out)  # batch * k
        mix_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h
        r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
                t.sum(mix_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
        return r

    def attention(self, u_vec, h_vecs):
        """
        这一层，我们可以尝试多种attention的方法：
        方案1：query 为user， key为历史tag, value同样为历史tag，然后将user+tag组成为新的user embedding
        方案2：query为user, key和value为tag进行一层MLP后的结果（从TransPITF的结果来看，进行MLP和激活后，或许可以得到一个好点的结果）
        :param u_vec: （batch_size, k)
        :param h_vecs (batch_size, m, k)
        :return:
        """
        # batch_size = u_vec.size()[0]
        # h_u_vec_ = self.relu(self.attentionMLP(u_vec))
        # u_vec_ = h_u_vec_.unsqueeze(2)
        # alpha = nn.functional.softmax(t.bmm(h_vecs, u_vec_).squeeze(2), 1)
        u_vec_ = u_vec.unsqueeze(2)        
        tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
        alpha = nn.functional.softmax(t.bmm(tag_h_vecs, u_vec_).squeeze(2), 1)
        
        alpha = alpha.unsqueeze(1)
        h = t.bmm(alpha, h_vecs)
        return h.squeeze(1)


class TagAttentionPITF(AttentionPITF):
    """
    直接基于attention机制的PITF, attention tag层面上进行attention
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """

    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma):
        super(TagAttentionPITF, self).__init__(numUser, numItem, numTag, k, init_st, m, gamma)
        self.tag_map = nn.Linear(2*k, k)

    def forward(self, x):
        """
        user与attention之后的h组合为新的user embedding或：
        直接将attention之后的向量作为user embedding
        :param x:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_vec_ids = x[:, 2]
        neg_tag_vec_ids = x[:, 3]
        history_ids = x[:, -self.m:]

        user_vecs = self.userVecs(user_vec_ids)
        item_vecs = self.itemVecs(item_vec_ids)
        user_tag_vecs = self.tagUserVecs(tag_vec_ids)
        item_tag_vecs = self.tagItemVecs(tag_vec_ids)
        neg_tag_user_vec = self.tagUserVecs(neg_tag_vec_ids)
        neg_tag_item_vec = self.tagItemVecs(neg_tag_vec_ids)
        tag_history_vecs = self.tagUserVecs(history_ids)

        h = self.attention(user_tag_vecs, tag_history_vecs)  # batch * k
        h_neg = self.attention(neg_tag_user_vec, tag_history_vecs)
        # mix_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h
        # mix_neg_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h_neg
        # r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
        #         t.sum(mix_neg_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
        user_tag_vecs_ = self.tag_map(t.cat((user_tag_vecs, h), 1))
        neg_tag_user_vecs_ = self.tag_map(t.cat((neg_tag_user_vec, h_neg), 1))
        r = t.sum(user_vecs * user_tag_vecs_, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
                t.sum(user_vecs * neg_tag_user_vecs_, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
        return r

    def attention(self, u_vec, h_vecs):
        """
        这一层，我们可以尝试多种attention的方法：
        方案1：query 为user， key为历史tag, value同样为历史tag，然后将user+tag组成为新的user embedding
        方案2：query为user, key和value为tag进行一层MLP后的结果（从TransPITF的结果来看，进行MLP和激活后，或许可以得到一个好点的结果）
        :param u_vec: （batch_size, k)
        :param h_vecs (batch_size, m, k)
        :return:
        """
        # batch_size = u_vec.size()[0]
        u_vec_ = u_vec.unsqueeze(2)
        tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
        alpha = nn.functional.softmax(t.bmm(tag_h_vecs, u_vec_).squeeze(2), 1)
        alpha = alpha.unsqueeze(1)
        h = t.bmm(alpha, h_vecs)
        return h.squeeze(1)

    def predict_top_k(self, x, k=5):
        """
        给定User 和 item  记忆序列，根据模型返回前k个tag
        输入sample:[u,i,m_1,m_2....m_j]
        :param u:
        :param i:
        :param k:
        :return:
        """
        numTag = len(self.tagItemVecs.weight)
        user_vec_ids = x[:, 0]
        user_vec_ids = user_vec_ids.repeat(numTag, 1)
        item_vec_ids = x[:, 1]
        item_vec_ids = item_vec_ids.repeat(numTag, 1)
        tag_memory_ids = x[:, -self.m:]
        tag_memory_ids = tag_memory_ids.repeat(numTag, 1)
        user_vec = self.userVecs(user_vec_ids).squeeze(1)
        # print(user_vec.size())
        # user_vec = user_vec.repeat(self.numTag, 1)
        item_vec = self.itemVecs(item_vec_ids).squeeze(1)
        h_vecs = self.tagUserVecs(tag_memory_ids)
        # h_vecs = h_vecs.repeat(self.numTag, 1)
        h = self.attention(self.tagUserVecs.weight, h_vecs)
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        user_tag_vecs = self.tag_map(t.cat((self.tagUserVecs.weight, h), 1))  # numTag * k
        #  print(mix_user_vec.size())
        # y = t.bmm(mix_user_vec.unsqueeze(1), self.tagUserVecs.weight.unsqueeze(2)) + t.bmm(item_vec.unsqueeze(
        #     1), self.tagItemVecs.weight.unsqueeze(2))
        y = t.bmm(user_vec.unsqueeze(1), user_tag_vecs.unsqueeze(2)) + t.bmm(item_vec.unsqueeze(1),
                                                                             self.tagItemVecs.weight.unsqueeze(2))
        y = t.squeeze(y)
        return y.topk(k)  # 按降序进行排列