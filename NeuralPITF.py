import torch as t
from torch.autograd import Variable
import torch.nn as nn
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
        return


class PITF_Loss(nn.Module):
    """
    定义PITF的loss function
    """
    def __init__(self):
        super(PITF_Loss, self).__init__()
        print("Use the BPR for Optimization")

    def forward(self, r_p, r_ne):
        return -t.log(t.sigmoid(r_p - r_ne))
