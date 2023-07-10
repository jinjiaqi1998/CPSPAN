import os, sys, random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler


class TrainDataset_Com(torch.utils.data.Dataset):
    def __init__(self, X_list, Y_list):
        self.X_list = X_list
        self.Y_list = Y_list
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        for v in range(self.view_size):
            current_x = self.X_list[v][index]
            current_x_list.append(current_x)
            current_y = self.Y_list[v][index]
            current_y_list.append(current_y)
        # X_list1 = self.X_list
        # Y_list1 = self.Y_list
        return current_x_list, current_y_list

    def __len__(self):
        # return the total size of data
        return self.X_list[0].shape[0]


class TrainDataset_All(torch.utils.data.Dataset):
    def __init__(self, X_list, Y_list, Miss_list):
        self.X_list = X_list
        self.Y_list = Y_list
        self.Miss_list = Miss_list
        self.view_size = len(X_list)

    def __getitem__(self, index):
        current_x_list = []
        current_y_list = []
        current_miss_list = []

        for v in range(self.view_size):
            current_x = self.X_list[v][index]
            current_x_list.append(current_x)
            current_y = self.Y_list[v][index]
            current_y_list.append(current_y)
            current_miss = self.Miss_list[v][index]
            current_miss_list.append(current_miss)
        # X_list1 = self.X_list
        # Y_list1 = self.Y_list
        return current_x_list, current_y_list, current_miss_list

    def __len__(self):
        # return the total size of data
        return self.X_list[0].shape[0]


class Data_Sampler(object):
    """Custom Sampler is required. This sampler prepares batch by passing list of
    data indices instead of running over individual index as in pytorch sampler"""

    def __init__(self, pairs, shuffle=False, batch_size=1, drop_last=False):
        if shuffle:
            self.sampler = RandomSampler(pairs)
        else:
            self.sampler = SequentialSampler(pairs)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = [batch]
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = [batch]
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


