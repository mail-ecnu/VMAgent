import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class PPOLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac
        self.params = list(mac.parameters())
        self.learn_cnt= 0
        self.optimiser = Adam(self.params, lr=args.lr)
        self.clip_grad_param = 1

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_param = list(self.target_mac.parameters())
        self.last_target_update_episode = 0
        self.tau = self.args.tau
        self.gpu_enable = True

        # PPO clip rate
        self.eps_clip = 0.2
    
    def _update_targets(self):
        for target_param, local_param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train(self, batch):
        gamma = 0.99

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
        a = th.LongTensor(action_list)#batch.action))
        rew = th.FloatTensor(reward_list)##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(mask_list)#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(next_obs)
        next_feat = th.FloatTensor(next_feat)
        next_avail = th.FloatTensor(next_avail)
        ind =  th.arange(a.shape[0])
        if self.gpu_enable:
            obs = obs.cuda()
            feat = feat.cuda()
            avail = avail.cuda()
            a = a.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()
            next_feat = next_feat.cuda()
            next_avail = next_avail.cuda()
            ind =  ind.cuda()
        # Calculate estimated Q-Values

        _, action_probs,log_pis, entropy = self.mac.select_actions([[obs, feat], avail])
        _, ola_action_probs,old_log_pis, old_entropy = self.target_mac.select_actions([[obs, feat], avail])

        ratios = th.exp(log_pis.cuda() - old_log_pis.cuda())

        # rew = (rew - rew.mean())/(rew.std() + 1e-10)
        
        Q = self.mac.agent.critic([obs,feat])
        V= (action_probs.cuda() * Q.cuda()).sum(1)
        advantages = (Q - V.unsqueeze(-1)).sum(1)

        surr1 = ratios * advantages
        surr2 = th.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

        # here are some hypreparameteres
        actor_loss = -th.min(surr1.cuda(), surr2.cuda())
        
        self.mac.agent.actor_optimizer.zero_grad()
        actor_loss.requires_grad_()
        actor_loss.mean().backward(retain_graph=True)
        self.mac.agent.actor_optimizer.step()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        with th.no_grad():
            _, action_probs_next,_, _ = self.mac.select_actions([[next_obs, next_feat], next_avail])
            Q_next = self.mac.agent.critic([next_obs,next_feat])
            V_next= (action_probs_next.cuda() * Q_next.cuda()).sum(1)

            target_v = rew + (self.args.gamma *(mask * V_next))
        # Compute critic loss
        critic_loss = 0.5 * F.mse_loss(V, target_v)

        # Update critics
        self.mac.agent.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.mac.agent.critic_optimizer.step()

        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.learn_cnt = 0

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    