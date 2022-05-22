import torch
import torch.nn as nn
import torch.nn.functional as F

ME = 60


def linear(epoch, pretrain=20):
    e = max(0, epoch - pretrain)
    return 1 / ME * e


class discretization_2operator(nn.Module):
    def __init__(self, weight=0.0, length=4):
        super(discretization_2operator, self).__init__()
        self.weight1 = weight
        self.weight2 = 0
        self.length = length

    def forward(self, model, epoch, pretrain=20, turns=0):
        self.update(epoch, pretrain)
        loss1 = 0
        loss2 = 0
        model.generate_weight()
        for weight in model.weight_edge:
            for k, W in enumerate(weight):
                if len(W) > 1 and k % (self.length + 1) == turns:
                    # print('loss k:', k)
                    avg = W.sum() / len(W)
                    for w in W:
                        loss1 += torch.log(w / avg + 1.)  # * w
        for W_OPs in model.weight_op:
            for k, W_OP in enumerate(W_OPs):
                for _, W in enumerate(W_OP):
                    if len(W) > 1 and k % (self.length + 1) == turns:
                        avg = W.sum() / len(W)
                        # print('avg', W)
                        for w in W:
                            loss2 += torch.log(w / avg + 1.)
        return self.weight1 * loss1, self.weight2 * loss2
        # return loss1, loss1.item(), 0

    def update(self, epoch, pretrain):
        self.weight1 = (linear(epoch, pretrain)) / 40
        self.weight2 = (linear(epoch, pretrain)) / 30


if __name__ == '__main__':
    weight = torch.tensor([[1, 1, 0.6, 0.6],
                           [1, 1, 0.6, 0.6]])
    l = 0
    for i in range(len(weight)):
        l += F.mse_loss(weight[i], torch.tensor(0.5, requires_grad=False))
    print(l)
