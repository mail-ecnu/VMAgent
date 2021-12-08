from collections import namedtuple
import random
import numpy as np
from typing import Deque, Dict, List, Tuple
Experience = namedtuple('Experience',
                        ('obs', 'action', 'reward', 'next_state', 'done'))

SMALL_REQ = [(1, 1), (1, 2), (1, 4), (4, 4), (2, 4)]


class ReplayMemory:
    def __init__(self, args):
        self.capacity = args.capacity
        self.memory = {'obs': np.array([]), 'feat': np.array([]), 'avail': np.array([]), 'action': np.array([]),
                       'reward': np.array([]), 'next_obs': np.array([]), 'next_feat': np.array([]), 'next_avail': np.array([]), 'done': np.array([])}
        self.nb_sampels = 0

    def clean(self,):
        self.memory = {'obs': np.array([]), 'feat': np.array([]), 'avail': np.array([]), 'action': np.array([]),
                       'reward': np.array([]), 'next_obs': np.array([]), 'next_feat': np.array([]), 'next_avail': np.array([]), 'done': np.array([])}
        self.nb_sampels = 0

    def push(self, exp):
        nb_added = exp['obs'].shape[0]
        if nb_added == 0:
            return
        self.nb_sampels += nb_added
        idxs = np.array([i % self.capacity for i in range(
            self.nb_sampels-nb_added, self.nb_sampels)])
        if self.nb_sampels > self.capacity:
            # self.memory.append(None)
            for key in self.memory.keys():
                self.memory[key][idxs] = exp[key]
        elif self.__len__() - nb_added == 0:
            for key in self.memory.keys():
                self.memory[key] = np.repeat(
                    np.array([exp[key][0]]), self.capacity, axis=0)
                self.memory[key][idxs] = exp[key]
        else:
            for key in self.memory.keys():
                self.memory[key][idxs] = exp[key]
    #         try:
    #             for key in self.memory.keys():
    # #                self.memory[key] = np.concatenate((self.memory[key], exp[key]), axis=0 )
    #                 self.memory[key][idxs] = exp[key]
    #         except:
    #             import pdb; pdb.set_trace()

    def anneal_bs(self, nb, k):
        cap_lst = [self.capacity*i//k for i in range(k)]
        i = 0
        for cap in cap_lst:
            i += 1
            if nb < cap or nb == cap:
                return i

    def sample(self, ori_batch_size):
        # import pdb;pdb.set_trace()
        # batch_size = self.anneal_bs(ori_batch_size, 4) * ori_batch_size
        # if batch_size > self.__len__():
        #     batch_size = self.__len__()//ori_batch_size * ori_batch_size
        batch_size = ori_batch_size
        idxs = np.array(random.sample(range(self.__len__()), batch_size))
        res = {}
        try:
            for key in self.memory.keys():
                res[key] = self.memory[key][idxs]
        except:
            import pdb
            pdb.set_trace()
        return res

    def __len__(self):
        if self.nb_sampels > self.capacity:
            return self.capacity
        else:
            return self.nb_sampels
        # return self.memory['obs'].shape[0]

    def can_sample(self, batch_size):
        return self.__len__() >= batch_size
