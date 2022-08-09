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
                # if key in ['avail','next_avail','done','action']:
                #     self.memory[key] = np.repeat(
                #         np.array([exp[key][0]]), self.capacity, axis=0)
                # else:
                #     self.memory[key] = np.repeat(
                #         np.float32([exp[key][0]]), self.capacity, axis=0)
                self.memory[key][idxs] = exp[key]
        else:
            for key in self.memory.keys():
                self.memory[key][idxs] = exp[key]

    def anneal_bs(self, nb, k):
        cap_lst = [self.capacity*i//k for i in range(k)]
        i = 0
        for cap in cap_lst:
            i += 1
            if nb < cap or nb == cap:
                return i

    def sample(self, ori_batch_size):
        batch_size = ori_batch_size
        if batch_size > self.__len__():
            batch_size = self.__len__()
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

    def can_sample(self, batch_size):
        return self.__len__() >= batch_size




class SumTree(object):# sumtree are the priority
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        self.memory = {'obs': np.array([]), 'feat': np.array([]), 'avail': np.array([]), 'action': np.array([]),
                       'reward': np.array([]), 'next_obs': np.array([]), 'next_feat': np.array([]), 'next_avail': np.array([]), 'done': np.array([])}
        self.nb_sampels = 0

    def add(self, p, exp):
        tree_idx = self.nb_sampels % self.capacity + self.capacity - 1
        if self.nb_sampels == 0:
            for key in self.memory.keys():
                self.memory[key] = np.repeat(
                    np.array([exp[key]]), self.capacity, axis=0)
                # if key in ['avail','next_avail','done','action']:
                #     self.memory[key] = np.repeat(
                #         np.array([exp[key]]), self.capacity, axis=0)
                # else:
                #     self.memory[key] = np.repeat(
                #         np.float32([exp[key]]), self.capacity, axis=0)
        else:
            for key in self.memory.keys():
                self.memory[key][self.nb_sampels % self.capacity] = exp[key]

        self.update(tree_idx, p)  # update tree_frame
        self.nb_sampels += 1

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # if p == 0.0:
        #     import pdb;pdb.set_trace()
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], {'obs': self.memory['obs'][data_idx], 'feat': self.memory['feat'][data_idx], \
            'avail': self.memory['avail'][data_idx], 'action': self.memory['action'][data_idx],'reward': self.memory['reward'][data_idx],\
            'next_obs': self.memory['next_obs'][data_idx], 'next_feat': self.memory['next_feat'][data_idx],\
            'next_avail': self.memory['next_avail'][data_idx], 'done': self.memory['done'][data_idx]}

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PriorityMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    

    def __init__(self, args):
        self.tree = SumTree(args.capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

    def clean(self,):
        self.tree.memory = {'obs': np.array([]), 'feat': np.array([]), 'avail': np.array([]), 'action': np.array([]),
                       'reward': np.array([]), 'next_obs': np.array([]), 'next_feat': np.array([]), 'next_avail': np.array([]), 'done': np.array([])}
        self.tree.nb_sampels = 0

    def push(self, exp):
        nb_added = exp['obs'].shape[0]
        if nb_added == 0:
            return

        for i in range(nb_added):
            buf = {'obs': exp['obs'][i], 'feat': exp['feat'][i], 'avail': exp['avail'][i], 'action': exp['action'][i],
               'reward': exp['reward'][i], 'next_obs': exp['next_obs'][i], 'next_feat': exp['next_feat'][i],
               'next_avail': exp['next_avail'][i], 'done': exp['done'][i]}
            max_p = np.max(self.tree.tree[-self.tree.capacity:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(max_p, buf)   # set the max p for new p    

    def sample(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()
        b_idx, ISWeights = np.empty((batch_size,), dtype=np.int64), np.empty((batch_size, 1))
        b_memory = {'obs': np.array([]), 'feat': np.array([]), 'avail': np.array([]), 'action': np.array([]),\
            'reward': np.array([]), 'next_obs': np.array([]), 'next_feat': np.array([]), 'next_avail': np.array([]), 'done': np.array([])}
        for key in b_memory.keys():
            b_memory[key] = np.repeat(
                    np.array([self.tree.memory[key][0]]), batch_size, axis=0)
            # if key in ['avail','next_avail','done','action']:
            #     b_memory[key] = np.repeat(
            #         np.array([self.tree.memory[key][0]]), batch_size, axis=0)
            # else:
            #     b_memory[key] = np.repeat(
            #         np.float32([self.tree.memory[key][0]]), batch_size, axis=0)

        pri_seg = self.tree.total_p / batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # min_prob = np.min(self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.__len__()]) / self.tree.total_p     # for later calculate ISweight
        # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, buf_sampled = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            
            ISWeights[i, 0] = np.power(prob * self.__len__(), -self.beta)
            for key in b_memory.keys():
                b_memory[key][i] = buf_sampled[key]
            b_idx[i] = idx
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)    

    def __len__(self):
        if self.tree.nb_sampels > self.tree.capacity:
            return self.tree.capacity
        else:
            return self.tree.nb_sampels