import torch.nn as nn
import torch.nn.functional as F
import torch as th
import torch.optim as optim

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class Actor(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(Actor, self).__init__()  
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

        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, self.num_actions),
            nn.ReLU(),
        )
        #self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        h = th.cat([h01, h11], dim = -1)

        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        h = th.cat([h01, h11], dim = -1)

        h= self.fc0(h)
        h3 = self.fc1(h)
        h3 = self.fc2(h3)

        action_probs = h3
        #action_probs = self.softmax(h3)

        return action_probs.detach().cpu()
    
           

class Critic(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(Critic, self).__init__()
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

        self.fc1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, self.obs_space[0]*2),
            nn.ReLU(),
        )

        self.value = nn.Linear(self.obs_space[0]*2, 1)
        self.adv = nn.Linear(self.obs_space[0]*2, self.num_actions)

    def forward(self, state):
        obs, feat = state[0], state[1]
        h01, h11 = self.flat(obs), self.flat(feat)
        h = th.cat([h01, h11], dim = -1)

        h= self.fc0(h)

        h3 = self.fc1(h)
        h3 = self.fc2(h3)

        adv = self.adv(h3)
        value = self.value(h3)

        q_values = value + adv - th.mean(adv, dim=1, keepdim=True)

        if len(q_values.shape) == 1:
            return q_values.reshape(1, -1)
        else:
            return q_values



class SACAgent(nn.Module):
    
    def __init__(self, state_space, act_space, args):
        super(SACAgent, self).__init__()
        self.obs_space, self.feat_space = state_space[0], state_space[1]
        self.num_actions = act_space
        
        self.gamma = args.gamma
        self.tau2 = 1e-2
        self.lr = args.lr
        self.clip_grad_param = 1

        self.target_entropy = -self.num_actions  # -dim(A)

        self.log_alpha = th.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.lr) 
                
        # Actor Network 

        self.actor_local = Actor(state_space, act_space, args).cuda()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_space, act_space, args).cuda()
        self.critic2 = Critic(state_space, act_space, args).cuda()
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_space, act_space, args).cuda()
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_space, act_space, args).cuda()
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr) 