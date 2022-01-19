import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from modules.critics import REGISTRY as critic_resigtry

class SACLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac
        self.learn_cnt= 0
        self.clip_grad_param = 1

        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry['ac_critic']([(args.N, 2, 2), (1, 2, 2)], args.N*2, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.q_critic = critic_resigtry['soft_q']([(args.N, 2, 2), (1, 2, 2)], args.N*2, args)
        self.target_q_critic = copy.deepcopy(self.q_critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.critic_q_params = list(self.q_critic.parameters())
        self.critic_q_optimiser = Adam(params=self.critic_q_params, lr=args.lr)

        self.log_alpha = th.tensor([0.0]).cuda()
        self.log_alpha.requires_grad = True
        self.alpha_optimiser = Adam(params=[self.log_alpha], lr=0.001)
        self.alpha = self.log_alpha.exp().detach() 

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        print(mac)
        self.alpha_update_step = 0
        self.gpu_enable = True
    
    def train(self, batch):
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
        
        mac_out, _ = self.mac.forward([[obs, feat], avail])
        pi = mac_out
        pi[mask == 0] = 1.0
        pi_taken = th.gather(pi, dim=1, index=actions.unsqueeze(-1))
        log_pi_taken = th.log(pi_taken + 1e-10)

        critic_mask = mask.clone()

        # Qcurrent  
        q_values = self.q_critic([obs,feat],actions)
        # Qtarget
        expected_q_values = self.target_q_critic([obs,feat],actions)
        # Vnext_traget
        target_values = self.target_critic([next_obs,next_feat])
        # Vcurrent
        expected_values = self.critic([obs,feat])

        # Qnet TD-error rew + γ*Vnext- Q
        next_q_values = rew + critic_mask * self.args.gamma * target_values        
        q_td_error = next_q_values.detach() - q_values
        masked_q_td_error = q_td_error * critic_mask

        #Vnet TD-error V - (Q-logpi)
        next_values = expected_q_values - log_pi_taken
        value_error = expected_values - next_values.detach()
        masked_value_error = value_error * critic_mask

        q_loss = (masked_q_td_error ** 2).sum() / mask.sum()
        critic_loss = (masked_value_error ** 2).sum() / mask.sum()

        self.critic_q_optimiser.zero_grad()
        q_loss.backward(retain_graph=True)
        grad_norm = clip_grad_norm_(self.critic_q_params, self.args.grad_norm_clip)
        self.critic_q_optimiser.step()

        self.critic_optimiser.zero_grad()
        critic_loss.backward(retain_graph=True)
        grad_norm = clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # Actor loss alpha*logpi - Q      
        actor_loss = ((self.alpha.detach() * log_pi_taken - expected_q_values) * mask).sum() / mask.sum()

        self.agent_optimiser.zero_grad()
        actor_loss.backward(retain_graph=True)
        grad_norm = clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        alpha_loss = - (self.log_alpha.exp() * (entropy + log_pi_taken).detach()).mean()

        self.alpha_update_step += 1
        if self.alpha_update_step % 5 == 0:
            self.alpha_optimiser.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

        self._update_targets_soft(self.args.tau)

        return {
            'actor_loss': actor_loss.item(),
            'q_critic_loss': q_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item()
        }

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_q_critic.parameters(), self.q_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.q_critic.cuda()
        self.target_q_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.q_critic.state_dict(), "{}/q_critic.th".format(path))
        th.save(self.alpha_optimiser.state_dict(), "{}/alpha_opt.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        th.save(self.critic_q_optimiser.state_dict(), "{}/q_critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        self.q_critic.load_state_dict(th.load("{}/q_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.alpha_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_q_optimiser.load_state_dict(th.load("{}/q_critic_opt.th".format(path), map_location=lambda storage, loc: storage))