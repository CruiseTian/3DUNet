
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        output = DiceAverage.get_froc(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        self.precision = output[0]
        self.recall = output[1]
        self.F1 = output[2]
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        logits = logits.sigmoid()
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

    @staticmethod
    def get_froc(logits, targets):
        # froc = []
        predict = np.array(list(np.where(logits.cpu().detach().numpy() < 0.5, 0, 1)[:])).astype(dtype=int)
        targets = np.array(targets.cpu().detach().numpy()).astype(dtype=int)

        TN = np.array(targets[predict == 0] == 0).sum()
        FP = np.array(targets[predict == 1] == 0).sum()
        FN = np.array(targets[predict == 0] == 1).sum()
        TP = np.array(targets[predict == 1] == 1).sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1
