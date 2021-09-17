import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import copy

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class QmixAgent(nn.Module):
    '''
        Virtual Done Agent: Assume the req has been placed
        TODO: regularize the virtual done(i.e. negative input to -1 to indicate not appliable)
    '''
    def __init__(self, state_space, act_space, args):
        super(QmixAgent, self).__init__()
        self.state_space = state_space[0]*args.N
        self.obs_space, self.feat_space = state_space[0], state_space[1]
        self.num_actions = act_space
        self.N = args.N
        self.abs_weight = (args.abs_weight=='True')

        self.flat = nn.Sequential(
            nn.Flatten(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.fc0 = nn.DataParallel(self.fc0)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.fc1 = nn.DataParallel(self.fc1)

        self.fc2 = nn.Sequential(
            nn.Linear(64, self.obs_space[0]*2),
            nn.ReLU(),
        )
        self.fc2 = nn.DataParallel(self.fc2)
        self.fc_o = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        self.fc_s = nn.Sequential(
            nn.Linear(4*self.N, 32),
            nn.ReLU()
        )

        self.fc_c = nn.Sequential(
            nn.Linear(16+32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.value = nn.Linear(self.obs_space[0]*2, 1)
        self.value = nn.DataParallel(self.value)
        self.adv = nn.Linear(self.obs_space[0]*2, self.num_actions)




    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        bs = h01.shape[0]
        h00 = copy.deepcopy(h01)
        # import pdb; pdb.set_trace()
        h00 = h00.repeat(1,self.N*2).reshape((-1,h01.shape[1]))
        h01 = h01.reshape((-1, 4)).repeat(1, 2).reshape(-1, 4)
        h11 = h11.repeat(1, self.N).reshape((-1, 4))
        # import pdb; pdb.set_trace()
        h = h01 - h11
        s = h00
        # import pdb; pdb.set_trace()
        s = self.fc_s(s)
        o = self.fc_o(h)
        os = th.cat([s,o], dim=-1)
        if self.abs_weight:
            # import pdb; pdb.set_trace()
            weights = th.abs(self.fc_c(os))
        else:
            weights = self.fc_c(os)

        h= self.fc0(h)

        h3 = self.fc1(h)
        h3 = self.fc2(h3)

        q_values = self.value(h3)
        # import ipdb; ipdb.set_trace()
        w_q_values = q_values * weights
        w_q_values = w_q_values.reshape((bs, -1))
        if len(w_q_values.shape) == 1:
            return w_q_values.reshape(1, -1)
        else:
            return w_q_values



