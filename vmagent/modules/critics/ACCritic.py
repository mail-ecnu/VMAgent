import torch.nn as nn
import torch.nn.functional as F
import torch as th

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


class ACCritic(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(ACCritic, self).__init__()
        self.obs_space, self.feat_space = state_space[0], state_space[1]
        self.num_actions = act_space

        self.flat = nn.Sequential(
            nn.Flatten(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(args.N * 4+8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.value = nn.Linear(256, 1)

    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        h = th.cat([h01, h11], dim = -1)

        h= self.fc0(h)

        value = self.value(h)

        return value
