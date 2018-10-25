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
        self.userTagTimeList = list()  # 用处用户使用tag的时间戳列表
        self.userShortMemory = dict()  # 记录预测用户历史序列
        self.alphaUser = 1.4
        self.alphaItem = 1.4
        self.data = data
        self.validation = validation
        self.is_timestamp = is_timestamp
        self.predictUserTagWeight = list()  # 预测时的user-tag 的权重值
        self.itemTagWeight = list()  # 根据 most popular 存储所有item-tag的权重
        self.userTagTrainWeight = list()
        """
        tao(u, tag, time) = initialTao（初始强度值）+ 该时间点之前的行为对当前行为的影响tao_star
        这个值是递归计算的tao_star
        """
        self.trainUserTagTimeTaoList = list()
        """
        sum_tao(u,tag,time) 归一化tao(u, tag, time)的分母
        """
        self.tempUserTimeSum = list()
        # 部分可调节的超参数
        self.d = 0.5
        self.base = 0
        self.userTagWeightNum = 0
        self.userTagWeightIterNum = 0
        self.initialTao = 1
        self.timeUnit = 24*3600*1000

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

        2018.10.16
        权重的计算：
        item-tag 是根据Most popular全局共享，并且并不随时间进行变化，因此只需要一个即可，正例负例读出即可
        user-tag： 初始化的时候，可以直接计算得出训练集所有正例在每个时间点的tag，负例需要用当前时间进行重新计算
                   对于测试集，所有user-tag的权重也可以事先计算好
        :return:
        """
        num_user, num_item, num_tag = self.calc_number_of_dimensions()
        item_tag_count_list = list()
        user_time_list = list()

        for u in range(num_user):
            self.userTagTimeList.append(dict())
            self.predictUserTagWeight.append(dict())
            self.userTagTrainWeight.append(dict())
            self.trainUserTagTimeTaoList.append(dict())
            self.tempUserTimeSum.append(dict())
            user_time_list.append(0)
        for i in range(num_item):
            self.itemTagWeight.append(dict())
            item_tag_count_list.append(dict())
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
            if tag not in item_tag_count_list[i].keys():
                item_tag_count_list[i][tag] = 0
            if tag not in self.userTagTimeList[u].keys():
                self.userTagTimeList[u][tag] = list()
            self.userTagTimeList[u][tag].append(time)
            self.userTimeList[u].append((i, tag, time))
            self.trainUserTagSet[u].add(tag)
            self.trainItemTagSet[i].add(tag)
            self.userTagTrainWeight[u][tag] = dict()
            item_tag_count_list[i][tag] += 1
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
                # 找到测试集中，用户打标签行为最晚的时间作为预测时间（存疑：为什么不直接用这次行为的时间？）
                user_time_list[u] = time if time > user_time_list[u] else user_time_list[u]
        self.cal_pre_user_tag_weights(num_user, user_time_list)  # 计算预测时的权重，同时计算训练集过程中的权重
        self.cal_train_user_tag_weights(num_user)  # 计算训练集中的user-tag权重，会用trainUserTagTimeList
        self.cal_item_tag_weights(num_tag, item_tag_count_list)

    def cal_pre_user_tag_weights(self, num_user, user_time_list):
        """
        计算预测时的 user_tag 权重
        :return:
        """
        # 对user在测试集中不同时间点使用的不同tag的时间权重进行归一化
        for u in range(num_user):
            normalize = 0
            predict_time = user_time_list[u]
            tags_time = self.userTagTimeList[u]
            for tag in tags_time.keys():
                temp_tags_time = tags_time[tag]
                temp_tags_time.sort()
                first_time = temp_tags_time[0]
                if tag not in self.trainUserTagTimeTaoList[u].keys():
                    self.trainUserTagTimeTaoList[u][tag] = dict()
                self.trainUserTagTimeTaoList[u][tag][first_time] = 0
                """
                对应 tao(u,tag,time) = initialTao + tao_star(u, tag, time)中的第二项
                对于第一个时间点，由于之前没有其他事件，所以tao_star = 0
                """
                tao_star = 0
                # 从列表的第二项开始计算到最后一项：
                for index in range(1, len(temp_tags_time)):
                    cur_time = temp_tags_time[index]
                    previous_time = temp_tags_time[index-1]
                    tao_star = (1+tao_star) * np.exp(-self.d*(cur_time - previous_time)/self.timeUnit)
                    self.trainUserTagTimeTaoList[u][tag][cur_time] = tao_star
                """
                 * 计算完递归的taoStar后，根据taoStar(lastTime)来计算tao(user,tag,predictTime)
                 * 当测试集中的predictTime和训练集中的lastTime相等时，我们仍然计算
                 * 结果为1+taoStar(lastTime)【加上了1这个常数值】,
                 * tao(user,tag,predictTime) = taoStar(lastTime) + initialTao
                """
                last_time = temp_tags_time[-1]
                # try:
                # print(-self.d*(predict_time-last_time)/self.timeUnit)
                # print([u, tag, last_time, predict_time])
                predict_tao = (1+self.trainUserTagTimeTaoList[u][tag][last_time]) * np.exp(-self.d*(predict_time-last_time)/self.timeUnit) + self.initialTao
                # except RuntimeWarning as e:
                #     print(-self.d*(predict_time-last_time)/self.timeUnit)
                #     print([u, tag, last_time, predict_time])
                #     raise
                normalize += predict_tao
                self.predictUserTagWeight[u][tag] = predict_tao
            for tag in tags_time.keys():
                value = 1 + np.log10(1+np.power(10, self.alphaUser)*self.predictUserTagWeight[u][tag]/normalize)
                self.predictUserTagWeight[u][tag] = value

    def cal_train_user_tag_weights(self, num_user):
        """
        计算训练集中, w_u,tag,time权重
        会利用前面计算好的trainUserTagTimeTaoStarList， 同时需要计算tempUserTimeSum，用户在每个时间点的用于归一化的tao的和
        对于用户行为中已经包含的时间点，直接取
        :param num_user:
        :param user_time_list:
        :return:
        """
        for u in range(num_user):
            tags_time = self.userTagTimeList[u]
            for tag in tags_time.keys():
                temp_tags_time = tags_time[tag]
                temp_tags_time.sort()
                for index in range(len(temp_tags_time)):
                    temp_time = temp_tags_time[index]
                    normalize = 0
                    normalize += self.trainUserTagTimeTaoList[u][tag][temp_time] + self.initialTao

                    # 再次遍历， 计算tao(u, tempTime)的和（对于每一个tag行为，需要将这个时间之前所有行为全部计算进行归一化）
                    for tag_key in tags_time.keys():
                        # 除去分子中的tag与其他tag对应的时间列表
                        if tag_key != tag:
                            each_time_list = tags_time[tag_key]
                            binary_index = self.binary_search(each_time_list, temp_time)
                            """
                            为计算tao(u, tagKey, tempTime),先找出tempTime前离得最近的时间点的索引binary_index
                            有三种情况：
                            1. 当tempTime存在于eachTimeList中时，直接取trainUserTagTimeTaoStarList.get(user).get(tagKey)
                            .get(tempTime)+initialTao即可
                            2. 找到了tempTime之前的索引，按公式计算
                            3. 找不到tempTime之前的索引，则tao值为0，无需累加normalizeSum
                            tao(u, tagKey, tempTime)=initialTao + taoStar
                            """
                            if binary_index != -1:
                                last_time = each_time_list[binary_index]
                                if last_time == temp_time:
                                    normalize += self.initialTao + self.trainUserTagTimeTaoList[u][tag_key][temp_time]
                                else:
                                    normalize += self.initialTao + np.exp(-self.d*(temp_time - last_time)/self.timeUnit)*(1+self.trainUserTagTimeTaoList[u][tag_key][last_time])
                    self.tempUserTimeSum[u][temp_time] = normalize
                    # if tag not in self.userTagTrainWeight[u].keys():
                    #     self.userTagTrainWeight[u][tag] = dict()
                    value = 0
                    if normalize == 0:
                        value = 1
                    else:
                        value = 1 + np.log10(1 + np.power(10, self.alphaUser) * (self.trainUserTagTimeTaoList[u][tag][temp_time]+self.initialTao)/normalize)
                    self.userTagTrainWeight[u][tag][temp_time] = value

    def cal_item_tag_weights(self, num_tag, item_tag_count_list):
        for item in range(num_tag):
            normalize = 0
            temp_tags_count = item_tag_count_list[item]
            for tag in temp_tags_count.keys():
                count = temp_tags_count[tag]
                normalize += count
                self.itemTagWeight[item][tag] = count
            for tag in temp_tags_count.keys():
                self.itemTagWeight[item][tag] = 1 + np.log10(1+np.power(10, self.alphaUser) * self.itemTagWeight[item][tag]/normalize)

    def binary_search(self, time_list, timestamp):
        """
        二分查找找到key之前的最大的时间戳
       *  在按时间排好序的用户的train时间序列上
       *  1.若key不包含在list中，返回key之前最大的时间戳，若找不到则返回-1
       * 2 .key本身就包含在list中，直接返回key在list中的索引
        :param time_list:
        :param timestamp:
        :return:
        """
        low = 0
        high = len(time_list) - 1
        while low < high:
            mid = int((low + high + 1)/2)
            if time_list[mid] > timestamp:
                high = mid - 1
            else:
                low = mid
        return low if time_list[low] <= timestamp else -1

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
            return u_max + 1, i_max + 1, t_max + 1
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

    def get_sequential(self, num_tag, m=5, num=10, weight=False):
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
            user_seqs = np.array(self.userTimeList[u])  # num * 3
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
                    neg_t = pairwise_sample[3]
                    # seq_data.append([u, item, tag, neg_t, timestamp, tag_memory, timestamp_memory])
                    if weight:
                        # print([u,tag,timestamp])
                        user_tag_weight = self.userTagTrainWeight[u][tag][timestamp]
                        item_tag_weight = self.get_item_weight(item, tag)
                        user_neg_tag_weight = self.get_neg_user_weight(u, neg_t, timestamp)
                        item_neg_tag_weight = self.get_item_weight(item, neg_t)
                        seq_data.append(pairwise_sample + [user_tag_weight, item_tag_weight, user_neg_tag_weight,
                                                           item_neg_tag_weight] + list(tag_memory) + [timestamp])
                    else:
                        seq_data.append(pairwise_sample + list(tag_memory) + [timestamp] + list(timestamp_memory))
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

    def get_neg_user_weight(self, u, tag, time):
        """
        根据输入数据，使用指数衰减，计算user-tag负例的权重
        :return:
        """
        weight = 0
        normalize = 0
        if tag in self.trainUserTagSet[u]:
            temp_tags_time_list = self.userTagTimeList[u][tag]
            binary_index = self.binary_search(temp_tags_time_list, time)
            if binary_index != -1:
                # time 是训练集中最小的时间戳时，tao为0
                if self.tempUserTimeSum[u][time] == 0:
                    weight = 1
                else:
                    last_time = temp_tags_time_list[binary_index]
                    if last_time == time:
                        # 在用户使用负例的tag时间中包含time,则可以直接取出已经算好的weight
                        weight = self.trainUserTagTimeTaoList[u][tag][time]
                    else:
                        # 在用户使用的负例tag时间序列中不包含time， time前最近的时间戳是lasttime
                        weight = (np.exp(-self.d*(time - last_time)/self.timeUnit) * (1+self.trainUserTagTimeTaoList[u][tag][last_time]) + self.initialTao) / self.tempUserTimeSum[u][time]
                        weight = 1 + np.log10(1+ np.power(10, self.alphaUser) * weight)
            else:
                weight = 1  # 用户在time之前没有用过tag,weight也为1
        else:
            weight = 1  # 用户在训练集中没有使用过tag，则权重为1
        return weight

    def get_item_weight(self, i, tag):
        """
        根据所有输入，计算most popular,作为item-tag权重, 注意，对于训练集中没有出现的tag，权重为1
        :return:
        """
        item_weight = 1
        if tag in self.itemTagWeight[i].keys():
            item_weight = self.itemTagWeight[i][tag]
        return item_weight

    def weight_to_vector(self, num_user, num_item, num_tag):
        """
        将predictUserTagWeight和itemTagWeight统一转为矩阵形式
        :return:
        """
        pre_user_weight = np.ones((num_user, num_tag))
        item_weight = np.ones((num_item, num_tag))
        for u in range(num_user):
            for tag in self.predictUserTagWeight[u].keys():
                pre_user_weight[u][tag] = self.predictUserTagWeight[u][tag]
        for i in range(num_item):
            for tag in self.itemTagWeight[i].keys():
                item_weight[i][tag] = self.itemTagWeight[i][tag]
        return pre_user_weight, item_weight


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
        return t.sum(-t.log(t.sigmoid(r)))# /len(r)


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
    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma, init_embeddings):
        super(TimeAttentionPITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.tagItemVecs = nn.Embedding(numTag, k, padding_idx=0)
        # self.attentionMLP = nn.Linear(k, k)
        self.user_tag_map = nn.Linear(4*k, k)
        self.relu = nn.ReLU()
        self.m = m
        self.k = k
        self.gamma = gamma
        self._init_weight(init_st, init_embeddings)

    def _init_weight(self, init_st, init_embedding):
        # self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        # self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        self.userVecs.weight.data = init_embedding[0]
        self.itemVecs.weight.data = init_embedding[1]
        self.tagUserVecs.weight.data[1:] = init_embedding[2]
        self.tagItemVecs.weight.data[1:] = init_embedding[3]

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
        add_vecs = user_vecs - h
        mul_vecs = user_vecs * h
        # mix_user_vecs = (1-self.gamma) * user_vecs + self.gamma * h
        mix_user_vecs = self.relu(self.user_tag_map(t.cat((user_vecs, h, add_vecs, mul_vecs), 1)))
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
        add_vec = user_vec - h
        mul_vec = user_vec * h
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        mix_user_vec = self.relu(self.user_tag_map(t.cat((user_vec, h, add_vec, mul_vec), 1)))
        y = mix_user_vec.mm(self.tagUserVecs.weight.t()) + item_vec.mm(self.tagItemVecs.weight.t())
        return y.topk(k)[1]  # 按降序进行排列


class AttentionPITF(nn.Module):
    """
    直接基于attention机制的PITF
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """
    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma, init_embedding):
        super(AttentionPITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.tagItemVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.attentionMLP = nn.Linear(k, k)
        self.user_tag_map = nn.Linear(4*k, k)
        self.relu = nn.ReLU()
        self.m = m
        self.k = k
        self.gamma = gamma
        self.dropout = nn.Dropout(0.3)
        self._init_weight(init_st, init_embedding)
        
    def _init_weight(self, init_st, init_embedding):
        self.userVecs.weight.data = init_embedding[0]
        self.itemVecs.weight.data = init_embedding[1]
        self.tagUserVecs.weight.data[1:] = init_embedding[2]
        self.tagItemVecs.weight.data[1:] = init_embedding[3]
        # self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        # self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
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

        user_vecs = self.dropout(self.userVecs(user_vec_ids))
        item_vecs = self.dropout(self.itemVecs(item_vec_ids))
        user_tag_vecs = self.dropout(self.tagUserVecs(tag_vec_ids))
        item_tag_vecs = self.dropout(self.tagItemVecs(tag_vec_ids))
        neg_tag_user_vec = self.dropout(self.tagUserVecs(neg_tag_vec_ids))
        neg_tag_item_vec = self.dropout(self.tagItemVecs(neg_tag_vec_ids))
        tag_history_vecs = self.dropout(self.tagUserVecs(history_ids))
        h = self.attention(user_vecs, tag_history_vecs)  # batch * k
        # mix_user_vecs = (1-self.gamma) * user_vecs + self.gamma * h
        add_vec = user_vecs - h
        mul_vec = user_vecs * h
        # mix_user_vecs = self.user_tag_map(t.cat((user_vecs, h, add_vec, mul_vec), 1))
        mix_user_vecs = self.relu(self.user_tag_map(t.cat((user_vecs, h, add_vec, mul_vec), 1)))
        # mix_user_vecs = self.relu(self.user_tag_map(t.cat((user_vecs, h), 1)))
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

        user_vec = self.dropout(self.userVecs(user_vec_ids))
        item_vec = self.dropout(self.itemVecs(item_vec_ids))
        h_vecs = self.dropout(self.tagUserVecs(tag_memory_ids))
        h = self.attention(user_vec, h_vecs)
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        # mix_user_vec = self.user_tag_map(t.cat((user_vec, h), 1))
        # mix_user_vec = self.relu(self.user_tag_map(t.cat((user_vec, h), 1)))
        add_vec = user_vec - h
        mul_vec = user_vec * h
        # mix_user_vec = self.user_tag_map(t.cat((user_vec, h, add_vec, mul_vec), 1))
        mix_user_vec = self.relu(self.user_tag_map(t.cat((user_vec, h, add_vec, mul_vec), 1)))
        y = mix_user_vec.mm(self.tagUserVecs.weight.t()) + item_vec.mm(self.tagItemVecs.weight.t())

        return y.topk(k)[1]  # 按降序进行排列


class RNNAttentionPITF(AttentionPITF):
    """
    基于RNN+attention机制的标签推荐模型
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """

    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma, init_embedding):
        super(RNNAttentionPITF, self).__init__(numUser, numItem, numTag, k, init_st, m, gamma, init_embedding)
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
        # mix_user_vecs = (1-self.gamma) * user_vecs + self.gamma * h
        add_vec = user_vecs - h
        mul_vec = user_vecs * h
        # mix_user_vecs = self.user_tag_map(t.cat((user_vecs, h, add_vec, mul_vec), 1))
        mix_user_vecs = self.relu(self.user_tag_map(t.cat((user_vecs, h, add_vec, mul_vec), 1)))
        # mix_user_vecs = self.relu(self.user_tag_map(t.cat((user_vecs, h), 1)))
        # mix_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h
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
        tag_h_vecs = h_vecs
        # tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
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
        # h = self.attention(user_vec, h_vecs)
        out, out_final = self.gru(h_vecs)
        h = self.attention(user_vec, out)  # batch * k
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        # mix_user_vec = self.user_tag_map(t.cat((user_vec, h), 1))
        # mix_user_vec = self.relu(self.user_tag_map(t.cat((user_vec, h), 1)))
        
        add_vec = user_vec - h
        mul_vec = user_vec * h
        mix_user_vec = self.relu(self.user_tag_map(t.cat((user_vec, h, add_vec, mul_vec), 1)))
        
        y = mix_user_vec.mm(self.tagUserVecs.weight.t()) + item_vec.mm(self.tagItemVecs.weight.t())

        return y.topk(k)[1]  # 按降序进行排列


class TagAttentionPITF(AttentionPITF):
    """
    直接基于attention机制的PITF, attention tag层面上进行attention
    输入数据为 [u,i,t,neg_t,m_1,m_2,m_3...]

    """

    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma, init_embeddings):
        super(TagAttentionPITF, self).__init__(numUser, numItem, numTag, k, init_st, m, gamma, init_embeddings)
        self.tag_map = nn.Linear(4*k, k)
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
        
        out, out_final = self.gru(tag_history_vecs)
        # out, out_final = self.gru(tag_history_vecs, user_vecs)
        h = self.attention(user_tag_vecs, out) # 注意该方法中，attention不使用一层Map
        h_neg = self.attention(user_tag_vecs, out)
        
        # h = self.attention(user_tag_vecs, tag_history_vecs)  # batch * k
        # h_neg = self.attention(neg_tag_user_vec, tag_history_vecs)
        # mix_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h
        # mix_neg_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h_neg
        # r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
        #         t.sum(mix_neg_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
        add_vec = user_tag_vecs - h
        mul_vec = user_tag_vecs * h
        user_tag_vecs_ = self.relu(self.tag_map(t.cat((user_tag_vecs, h, add_vec, mul_vec), 1)))
        add_vec = neg_tag_user_vec - h_neg
        mul_vec = neg_tag_user_vec * h_neg
        neg_tag_user_vecs_ = self.relu(self.tag_map(t.cat((neg_tag_user_vec, h_neg,add_vec, mul_vec), 1)))
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
        tag_h_vecs = h_vecs
        # tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
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
        
        out, out_final = self.gru(h_vecs)
        # out, out_final = self.gru(tag_history_vecs, user_vecs)
        h = self.attention(self.tagUserVecs.weight, out) # 注意该方法中，attention不使用一层Map
        
        # h = self.attention(self.tagUserVecs.weight, h_vecs)
        
        add_vec = self.tagUserVecs.weight - h
        mul_vec = self.tagUserVecs.weight * h
        user_tag_vecs = self.relu(self.tag_map(t.cat((self.tagUserVecs.weight, h, add_vec, mul_vec), 1)))  # numTag * k
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        #  print(mix_user_vec.size())
        # y = t.bmm(mix_user_vec.unsqueeze(1), self.tagUserVecs.weight.unsqueeze(2)) + t.bmm(item_vec.unsqueeze(
        #     1), self.tagItemVecs.weight.unsqueeze(2))
        y = t.bmm(user_vec.unsqueeze(1), user_tag_vecs.unsqueeze(2)) + t.bmm(item_vec.unsqueeze(1),
                                                                             self.tagItemVecs.weight.unsqueeze(2))
        y = t.squeeze(y)
        return y.topk(k)[1]  # 按降序进行排列


class AttentionTAPITF(nn.Module):
    """
    这个模型基本设计：
    1·使用TAPITF的思路，user-tag和item-tag上分别添加权重
    2·将每次的tag作为query,衡量tag与过去行为之间的相似性
    权重计算的原则：
    1.只考虑user-tag，不考虑多元的影响
    2.我们在数据预处理中计算所有可能需要的权重，包括：
        每个用户所有使用tag的时间序列权重
        最流行的tag的时间序列权重
    我们把权重的计算放在数据预处理这一块，将权重作为输入数据，直接传入模型
    由于训练数据的权重可以由JAVA代码直接计算，那么我们数据预处理需要计算的内容仅仅是每次所生成负例的权重

    我们将各种情况实现在一个类中，通过多余的参数进行控制：
    1. 直接使用TAPITF
    2. 实现user query 的attention (权重和映射都可以试一试，映射更加有说服力）
    3. 实现tag query 的 attention

    最后，我们修改最后一层逻辑，参考神经协同过滤的思想，使用多种思路组合user_vec,user_tag_vec,item_vec,item_tag_vec
    1. 直接四维向量进行拼接，得到最后的结果（这种方法最后时限，因为相对不具备很好的解释性）
    2. user与tag 进行一层全连接， item与tag进行一层全连接
    3. 将传统点击和MLP学习到的特征，再进行一次拼接
    在每层的搭建上，使用塔的结果，每一层减少一半的规模
    """
    def __init__(self, numUser, numItem, numTag, k, init_st, m, gamma, init_embeddings, user_weights, item_weights, use_attention=True, query_type='tag', cf_type='t-MLP'):
        super(AttentionTAPITF, self).__init__()
        self.userVecs = nn.Embedding(numUser, k)
        self.itemVecs = nn.Embedding(numItem, k)
        self.tagUserVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.tagItemVecs = nn.Embedding(numTag, k, padding_idx=0)
        self.attentionMLP = nn.Linear(k, k)
        self.user_tag_map = nn.Linear(4*k, k)
        self.relu = nn.ReLU()
        self.m = m
        self.k = k
        self.gamma = gamma
        self.user_weights = user_weights  # numTag * 1
        self.item_weights = item_weights  # numTag * 1
        self._init_weight(init_st, init_embeddings)
        self.tag_map = nn.Linear(4*k, k)
        self.use_attention = use_attention  # 该参数控制是否使用 attention 机制
        self.query_type = query_type  # 该参数控制 user 还是tag query（默认为tag)
        self.cf_type = cf_type
        if self.cf_type == 't-MLP':
            # 先尝试只用三层
            self.user_tag_mlp = nn.Linear(2*k, k)
            self.item_tag_mlp = nn.Linear(2*k, k)
            self.user_item_tag_mlp = nn.Linear(2*k, 1)
        elif self.cf_type == 'TAMLP':
            self.user_tag_mlp = nn.Linear(2*k, 1)
            self.item_tag_mlp = nn.Linear(2*k, 1)

    def _init_weight(self, init_st, init_embedding):
        # self.userVecs.weight = nn.init.normal(self.userVecs.weight, 0, init_st)
        # self.itemVecs.weight = nn.init.normal(self.itemVecs.weight, 0, init_st)
        self.userVecs.weight.data = init_embedding[0]
        self.itemVecs.weight.data = init_embedding[1]
        self.tagUserVecs.weight.data[1:] = init_embedding[2]
        self.tagItemVecs.weight.data[1:] = init_embedding[3]

    def forward(self, x):
        """
        首先，使用tag与tag_history进行attention，组合成新的tag embedding
        还可以user与attention之后的h组合为新的user embedding或：
        [u,i,t,neg_t,user_weight, item_weight, neg_user_weight, neg_item_weight, m_1,m_2,m_3...]
        :param x:
        :return:
        """
        user_vec_ids = x[:, 0]
        item_vec_ids = x[:, 1]
        tag_vec_ids = x[:, 2]
        neg_tag_vec_ids = x[:, 3]
        user_weight = x[:, 4]
        item_weight = x[:, 5]
        neg_user_weight = x[:, 6]
        neg_item_weight = x[:, 7]
        # user_weight = t.FloatTensor(user_weight).cuda()
        # item_weight = t.FloatTensor(item_weight).cuda()
        # neg_user_weight = t.FloatTensor(neg_user_weight).cuda()
        # neg_item_weight = t.FloatTensor(neg_item_weight).cuda()
        history_ids = x[:, -self.m:]

        user_vecs = self.userVecs(user_vec_ids)
        item_vecs = self.itemVecs(item_vec_ids)
        user_tag_vecs = self.tagUserVecs(tag_vec_ids)
        item_tag_vecs = self.tagItemVecs(tag_vec_ids)
        neg_tag_user_vec = self.tagUserVecs(neg_tag_vec_ids)
        neg_tag_item_vec = self.tagItemVecs(neg_tag_vec_ids)
        tag_history_vecs = self.tagUserVecs(history_ids)

        if self.use_attention:
            # out, out_final = self.gru(tag_history_vecs)
            # out, out_final = self.gru(tag_history_vecs, user_vecs)
            # h = self.attention(user_tag_vecs, out)  # 注意该方法中，attention不使用一层Map
            # h_neg = self.attention(user_tag_vecs, out)
            if self.query_type == 'tag':
                h = self.attention(user_tag_vecs, tag_history_vecs)  # batch * k
                h_neg = self.attention(neg_tag_user_vec, tag_history_vecs)
                # mix_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h
                # mix_neg_user_vecs = (1 - self.gamma) * user_vecs + self.gamma * h_neg
                # r = t.sum(mix_user_vecs * user_tag_vecs, dim=1) + t.sum(item_vecs * item_tag_vecs, dim=1) - (
                #         t.sum(mix_neg_user_vecs * neg_tag_user_vec, dim=1) + t.sum(item_vecs * neg_tag_item_vec, dim=1))
                add_vec = user_tag_vecs - h
                mul_vec = user_tag_vecs * h
                user_tag_vecs = self.relu(self.tag_map(t.cat((user_tag_vecs, h, add_vec, mul_vec), 1)))
                add_vec = neg_tag_user_vec - h_neg
                mul_vec = neg_tag_user_vec * h_neg
                neg_tag_user_vec = self.relu(self.tag_map(t.cat((neg_tag_user_vec, h_neg, add_vec, mul_vec), 1)))
                # w_u = user_weight.float() * t.sum(user_vecs * user_tag_vecs, dim=1)
                # w_i = item_weight.float() * t.sum(item_vecs * item_tag_vecs, dim=1)
                # w_neg_u = neg_user_weight.float() * t.sum(user_vecs * neg_tag_user_vec, dim=1)
                # w_neg_i = neg_item_weight.float() * t.sum(item_vecs * neg_tag_item_vec, dim=1)
                # r = w_u + w_i - (w_neg_u + w_neg_i)
            else:
                h = self.attention(user_vecs, tag_history_vecs)
                add_vec = user_vecs - h
                mul_vec = user_vecs * h
                user_vecs = self.relu(self.tag_map(t.cat((user_vecs, h, add_vec, mul_vec), 1)))
                # w_u = user_weight.float() * t.sum(user_vecs * user_tag_vecs, dim=1)
                # w_i = item_weight.float() * t.sum(item_vecs * item_tag_vecs, dim=1)
                # w_neg_u = neg_user_weight.float() * t.sum(user_vecs * neg_tag_user_vec, dim=1)
                # w_neg_i = neg_item_weight.float() * t.sum(item_vecs * neg_tag_item_vec, dim=1)
                # r = w_u + w_i - (w_neg_u + w_neg_i)

        if self.cf_type == 'GMF':
            w_u = user_weight.float() * t.sum(user_vecs * user_tag_vecs, dim=1)
            w_i = item_weight.float() * t.sum(item_vecs * item_tag_vecs, dim=1)
            w_neg_u = neg_user_weight.float() * t.sum(user_vecs * neg_tag_user_vec, dim=1)
            w_neg_i = neg_item_weight.float() * t.sum(item_vecs * neg_tag_item_vec, dim=1)
            r = w_u + w_i - (w_neg_u + w_neg_i)
        elif self.cf_type == 'MLP':
            r = 0
        elif self.cf_type == 't-MLP':
            # 先不加时间权重试试（应该效果难说）
            w_u = self.relu(self.user_tag_mlp(t.cat((user_vecs, user_tag_vecs), 1)))
            w_i = self.relu(self.item_tag_mlp(t.cat((item_vecs, item_tag_vecs), 1)))
            w_neg_u = self.relu(self.user_tag_mlp(t.cat((user_vecs, neg_tag_user_vec), 1)))
            w_neg_i = self.relu(self.item_tag_mlp(t.cat((item_vecs, neg_tag_item_vec), 1)))
            r = t.sigmoid(self.user_item_tag_mlp(t.cat((w_u, w_i), 1))) - t.sigmoid(self.user_item_tag_mlp(t.cat((w_neg_u, w_neg_i), 1)))
            # r =self.relu(self.user_item_tag_mlp(t.cat((w_u, w_i), 1))) -self.relu(self.user_item_tag_mlp(t.cat((w_neg_u, w_neg_i), 1)))
        elif self.cf_type == 'TAMLP':
            # user_tag 和 item_tag de 的向量进过MLP拼接后，加入权重再进行组合
            w_u = self.relu(self.user_tag_mlp(t.cat((user_vecs, user_tag_vecs), 1)))
            w_i = self.relu(self.item_tag_mlp(t.cat((item_vecs, item_tag_vecs), 1)))
            w_neg_u = self.relu(self.user_tag_mlp(t.cat((user_vecs, neg_tag_user_vec), 1)))
            w_neg_i = self.relu(self.item_tag_mlp(t.cat((item_vecs, neg_tag_item_vec), 1)))
            r = user_weight.float()* w_u + item_weight.float() * w_i - (neg_user_weight.float() * w_neg_u + neg_item_weight.float() * w_neg_i)
        else:
            r = 0
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
        tag_h_vecs = h_vecs
        # if self.query_type == 'user':
        tag_h_vecs = self.relu(self.attentionMLP(h_vecs))
        alpha = nn.functional.softmax(t.bmm(tag_h_vecs, u_vec_).squeeze(2), 1)
        alpha = alpha.unsqueeze(1)
        h = t.bmm(alpha, h_vecs)
        return h.squeeze(1)

    def predict_top_k(self, x, k=5):
        """
        给定User 和 item  记忆序列，根据模型返回前k个tag
        输入sample:[u,i, user_weight, item_weight, m_1,m_2....m_j]

        这里注意，测试用例的权重是所有的tag都需要计算，因为事先训练好，可以作为模型一个预训练的常量
        item_weight可以直接是Most_popluar
        user_weight则
        :return:
        """
        numTag = len(self.tagItemVecs.weight)
        user_vec_ids = x[:, 0]
        user_weight = t.FloatTensor(self.user_weights[user_vec_ids]).cuda()
        user_vec_ids = user_vec_ids.repeat(numTag, 1)
        item_vec_ids = x[:, 1]
        item_weight = t.FloatTensor(self.item_weights[item_vec_ids]).cuda()
        item_vec_ids = item_vec_ids.repeat(numTag, 1)
        tag_memory_ids = x[:, -self.m:]
        tag_memory_ids = tag_memory_ids.repeat(numTag, 1)
        user_vec = self.userVecs(user_vec_ids).squeeze(1)
        # print(user_vec.size())
        # user_vec = user_vec.repeat(self.numTag, 1)
        item_vec = self.itemVecs(item_vec_ids).squeeze(1)
        h_vecs = self.tagUserVecs(tag_memory_ids)
        # h_vecs = h_vecs.repeat(self.numTag, 1)
        user_tag_vecs = self.tagUserVecs.weight
        if self.use_attention:
            if self.query_type == 'tag':
                # out, out_final = self.gru(h_vecs)
                # out, out_final = self.gru(tag_history_vecs, user_vecs)
                # h = self.attention(self.tagUserVecs.weight, out)  # 注意该方法中，attention不使用一层Map

                h = self.attention(self.tagUserVecs.weight, h_vecs)

                add_vec = self.tagUserVecs.weight - h
                mul_vec = self.tagUserVecs.weight * h
                user_tag_vecs = self.relu(self.tag_map(t.cat((self.tagUserVecs.weight, h, add_vec, mul_vec), 1)))  # numTag * k

            else:
                h = self.attention(user_vec, h_vecs)
                add_vec = self.tagUserVecs.weight - h
                mul_vec = self.tagUserVecs.weight * h
                user_vec = self.relu(
                    self.tag_map(t.cat((user_vec, h, add_vec, mul_vec), 1)))  # numTag * k
        # mix_user_vec = (1 - self.gamma) * user_vec + self.gamma * h
        #  print(mix_user_vec.size())
        # y = t.bmm(mix_user_vec.unsqueeze(1), self.tagUserVecs.weight.unsqueeze(2)) + t.bmm(item_vec.unsqueeze(
        #     1), self.tagItemVecs.weight.unsqueeze(2))
        user_tag = t.bmm(user_vec.unsqueeze(1), user_tag_vecs.unsqueeze(2))
        item_tag = t.bmm(item_vec.unsqueeze(1), self.tagItemVecs.weight.unsqueeze(2))
        y = user_weight * user_tag.squeeze() + item_weight * item_tag.squeeze()
        # y = t.squeeze(y)
        # print(y.topk(k))
        if self.cf_type == 't-MLP':
            user_tag = self.relu(self.user_tag_mlp(t.cat((user_vec, user_tag_vecs), 1)))
            item_tag = self.relu(self.item_tag_mlp(t.cat((item_vec, self.tagItemVecs.weight), 1)))
            y = t.sigmoid(self.user_item_tag_mlp(t.cat((user_tag, item_tag), 1)))
            # y = self.relu(self.user_item_tag_mlp(t.cat((user_tag, item_tag), 1)))
            y = t.squeeze(y)
        elif self.cf_type == 'TAMLP':
            user_tag = self.relu(self.user_tag_mlp(t.cat((user_vec, user_tag_vecs), 1)))
            item_tag = self.relu(self.item_tag_mlp(t.cat((item_vec, self.tagItemVecs.weight), 1)))
            y = user_weight * user_tag.squeeze() + item_weight * item_tag.squeeze()
        return y.topk(k)[1]  # 按降序进行排列
