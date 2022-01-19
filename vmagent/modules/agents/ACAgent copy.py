import torch.nn as nn
import torch.nn.functional as F
import torch as th

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class ACAgent(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(ACAgent, self).__init__()  
        self.obs_space, self.feat_space = state_space[0], state_space[1]
        self.num_actions = act_space

        self.flat = nn.Sequential(
            nn.Flatten(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(args.N * 4+8, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, self.num_actions),
            nn.Tanh(),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        h = th.cat([h01, h11], dim = -1)

        h= self.fc0(h)
        h3 = self.fc1(h)
        action_probs = self.fc2(h3)

        # action_probs = self.softmax(action_probs)
        
        return action_probs
    
           