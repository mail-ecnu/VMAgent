import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class DQNAgent_modify(nn.Module):
    '''
        Virtual Done Agent: Assume the req has been placed
        TODO: regularize the virtual done(i.e. negative input to -1 to indicate not appliable)
    '''
    def __init__(self, state_space, act_space, args):
        super(DQNAgent_modify, self).__init__()
        self.obs_space, self.feat_space = state_space[0], state_space[1]
        self.num_actions = act_space
        self.N = args.N

        self.flat = nn.Sequential(
            nn.Flatten(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(args.N * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_space[0]*2),
            nn.ReLU(),
        )

        self.value = nn.Linear(self.obs_space[0]*2, 1)

    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        q_before = self.value(self.fc2(self.fc1(self.fc0(h01))))

        bs = h01.shape[0]

        nn_input = []

        for j in range(bs):
            current_cluster = obs[j]
            current_feat = feat[j]

            for i in range(self.N):
                current_server = current_cluster.clone()
                current_server[i] -= current_feat[0]
                
                after_server = th.flatten(current_server,0)
                nn_input.append(after_server.tolist())

                current_server_2 = current_cluster.clone()
                current_server_2[i] -= current_feat[1]
                after_server = th.flatten(current_server_2,0)
                nn_input.append(after_server.tolist())
        
        nn_input = th.tensor(nn_input,device='cuda')
        nn_input /= th.tensor([42 if i%2==0 else 160 for i in range(len(nn_input[0]))],device='cuda')

        h= self.fc0(nn_input)

        h3 = self.fc1(h)
        h3 = self.fc2(h3)

        q_after = self.value(h3)
        q_after = q_after.reshape(bs,-1)
        q_values = q_after - q_before

        return q_values.reshape(bs,-1)
