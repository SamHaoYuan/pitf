from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class InputToVector(nn.Module):
    """
    实现第一层，根据user_id, item_id 与 tag_id

    """
    def __init__(self, ):
        super(InputToVector, self).__init__()

class NeuralPITF():
    """
    使用Pytorch，基于神经网络的思想实现PITF，
    """
    def __init__(self):
        super(NeuralPITF, self).__init__()
        self.embedding = nn.Embedding()
