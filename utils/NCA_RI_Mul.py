import torch
from torch import nn
from torch.autograd import Function
import math

eps = 1e-8

class NCA_RI_Add_CrossEntropy(nn.Module): 

    def __init__(self, clsLabels, insLabels, lambda_=0.1, margin=0):
        super().__init__()
        # register a buffer
        self.register_buffer('clsLabels', torch.LongTensor(clsLabels.size(0)))
        self.register_buffer('insLabels', torch.LongTensor(insLabels.size(0)))
        # set the buffer
        self.clsLabels = clsLabels
        self.insLabels = insLabels
        self.margin = margin
        self.lambda_ = lambda_

    def forward(self, x, indexes):

        batchSize = x.size(0)
        # memory size
        n = x.size(1)
        exp = torch.exp(x)

        # cls labels for currect batch
        cls_y = torch.index_select(self.clsLabels, 0, indexes.data).view(batchSize, 1)
        cls_same = cls_y.repeat(1, n).eq_(self.clsLabels)

        # ins labels for current batch
        ins_y = torch.index_select(self.insLabels, 0, indexes.data).view(batchSize, 1)
        ins_same = ins_y.repeat(1, n).eq_(self.insLabels)

        # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0) # P_ii=0
        Z = exp.sum(dim=1)

        p1 = torch.mul(exp, cls_same.float()).sum(dim=1)
        prob1 = torch.div(p1, Z)

        p2 = torch.mul(exp, ins_same.float()).sum(dim=1)
        prob2 = torch.div(p2, Z)

        prob1_masked = torch.masked_select(prob1, prob1.ne(0))
        prob2_masked = torch.masked_select(prob2, prob2.ne(0))

        clsLoss = - prob1_masked.log().sum(0) / batchSize
        insLoss = - self.lambda_ * prob2_masked.log().sum(0) / batchSize

        return clsLoss, insLoss
