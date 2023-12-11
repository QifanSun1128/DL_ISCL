import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch import nn


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent

class ConLoss(nn.Module):
    """Contrastive Learning: """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, group_source, group_target):
        """Compute loss for model. 
        adapted from Supervised Contrastive Learning:
        Args:
            group_source: source dictionary 
            group_target: target dictionary.
        Returns:
            A loss scalar.
        """ 
        loss = torch.tensor(0.00)
        z_a_target =[]
        z_a_source =[]
        for key in group_source.keys():
            if key in group_target:
                z_a_source.append(torch.stack(group_source[key])).cuda()
                z_a_target.append(torch.stack(group_target[key])).cuda()
            
        z_a_target = torch.cat(z_a_target, dim = 0).cuda() # dimension: number of unlabeled target sample x number of class
        z_a_source = torch.cat(z_a_source, dim = 0).cuda() # dimension: number of unlabeled source sample x number of class
        z_a = torch.cat([z_a_target, z_a_source], dim=0).cuda() #combine z_a_target and z_a_source to create a matrix of all samples that has dimension: (T x k)
        #jacky你懂吧 不需要comment了 ^_^
        for k in group_source.keys():
            Z_j = torch.stack(group_source[k])
            if k in group_target.keys():
                for z_i in group_target[k]:
                    den = torch.exp(torch.matmul(z_a, z_i.T) / self.temperature).sum() - torch.exp(torch.dot(z_i, z_i) / self.temperature)
                    num = torch.exp(torch.matmul(Z_j, z_i.T) / self.temperature)
                    log_prob = - torch.mean(torch.log(num/den))
                    loss += log_prob

        return loss
