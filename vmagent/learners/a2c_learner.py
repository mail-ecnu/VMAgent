import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from modules.critics import REGISTRY as critic_resigtry

class A2CLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac
        self.learn_cnt= 0
        self.clip_grad_param = 1

        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry['ac_critic']([(args.N, 2, 2), (1, 2, 2)], args.N*2, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        print(mac)
        self.gpu_enable = True
    
    def train(self, batch, train_order, priority_flag=False, ISWeights=[]):
        # Get the relevant quantities
        obs = batch['obs']
        feat = batch['feat']
        avail = batch['avail']
        action_list = batch['action']
        reward_list = batch['reward']
        next_obs = batch['next_obs']
        next_feat = batch['next_feat']
        mask_list = 1 - batch['done']
        next_avail = batch['next_avail']
        y_list = [0]

        obs = th.FloatTensor(obs)
        feat = th.FloatTensor(feat)
        avail = th.FloatTensor(avail)
        actions = th.LongTensor(action_list)#batch.action))
        rew = th.FloatTensor(reward_list)##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(mask_list)#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(next_obs)
        next_feat = th.FloatTensor(next_feat)
        next_avail = th.FloatTensor(next_avail)
        ind =  th.arange(actions.shape[0])
        if self.gpu_enable:
            obs = obs.cuda()
            feat = feat.cuda()
            avail = avail.cuda()
            actions = actions.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()
            next_feat = next_feat.cuda()
            next_avail = next_avail.cuda()
            ind =  ind.cuda()
        
        # standardise rewards
        # rew = (rew - rew.mean()) / (rew.std() + 1e-5)
        
        # Q = rewards + γ * v_next
        rew = rew.unsqueeze(-1)
        critic_mask = mask.unsqueeze(-1)
        v_next = self.target_critic([next_obs,next_feat])
        target_vals = rew + (self.args.gamma * critic_mask * v_next)
        vals = self.critic([obs, feat])
    
        td_error = (target_vals.detach() - vals)
        masked_td_error = td_error * critic_mask
        critic_loss = (masked_td_error ** 2).sum() / critic_mask.sum()

        if train_order == 0:
            self.critic_optimiser.zero_grad()
        critic_loss.backward()
        grad_norm = clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        if train_order+1 == self.args.train_n:
            self.critic_optimiser.step()

        mac_out, _ = self.mac.forward([[obs, feat], avail])
        pi = mac_out
        advantages = masked_td_error.detach()

        pi_taken = th.gather(pi, dim=1, index=actions.unsqueeze(-1))
        log_pi_taken = th.log(pi_taken + 1e-10)

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        entropy[mask==0] = 0.0
        entropy = entropy.unsqueeze(-1)
        
        actor_loss = -((advantages * log_pi_taken+self.args.entropy_coef * entropy) * critic_mask).sum() / critic_mask.sum()

        if train_order == 0:
            self.agent_optimiser.zero_grad()
        actor_loss.backward()
        grad_norm = clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        if train_order+1 == self.args.train_n:
            self.agent_optimiser.step()

        self.learn_cnt += 1

        if self.args.tau < 1:
            self._update_targets_soft(self.args.tau)
        elif  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.learn_cnt = 0
       
        if priority_flag == True:
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
            },td_error
        else:
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
            }

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path, x):
        self.mac.save_models(path, x)
        th.save(self.critic.state_dict(), f"{path}/critic_epoch{x}.th")
        th.save(self.agent_optimiser.state_dict(), f"{path}/agent_opt_epoch{x}.th")
        th.save(self.critic_optimiser.state_dict(), f"{path}/critic_opt_epoch{x}.th")

    def load_models(self, path, x):
        self.mac.load_models(path, x)
        self.critic.load_state_dict(th.load(f"{path}/critic_epoch{x}.th", map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load(f"{path}/agent_opt_epoch{x}.th", map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load(f"{path}/critic_opt_epoch{x}.th", map_location=lambda storage, loc: storage))
    