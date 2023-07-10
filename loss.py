import torch
import torch.nn as nn

class Proto_Align_Loss(nn.Module):
    def __init__(self):
        super(Proto_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss1 = mse(gt, P)

        return Loss1


class Instance_Align_Loss(nn.Module):
    def __init__(self):
        super(Instance_Align_Loss, self).__init__()

    def forward(self, gt, P):
        mse = nn.MSELoss()
        Loss2 = mse(gt, P)

        return Loss2



