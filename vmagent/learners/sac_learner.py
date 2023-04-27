import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class SACLearner:
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

    def soft_update(self, local_model , target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def _update_targets(self):
        for target_param, local_param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
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

        mac_out, _ = self.mac.forward([[obs, feat], avail])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], next_avail])[0]

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.mac.agent.alpha)

        _, action_probs, log_pis = self.mac.get_act_probs([[obs, feat], avail])

        q1 = self.mac.agent.critic1([obs, feat])
        V1 = (action_probs.cuda() * q1.cuda()).sum(1)
        q2 = self.mac.agent.critic2([obs, feat])
        V2 = (action_probs.cuda() * q2.cuda()).sum(1)

        min_Q = action_probs.cuda() * th.min(q1,q2)

        actor_loss = (self.mac.agent.alpha.cuda() * log_pis.cuda() - min_Q).sum(1).mean()
        
        
        self.mac.agent.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.mac.agent.actor_optimizer.step()
        
        # Compute alpha loss
        entropy = (log_pis * action_probs).sum(1)
        alpha_loss = - (self.mac.agent.log_alpha.exp() * (entropy.cpu() + self.mac.agent.target_entropy).detach().cpu()).mean()
        self.mac.agent.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.mac.agent.alpha_optimizer.step()
        self.mac.agent.alpha = self.mac.agent.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with th.no_grad():
            idx = th.eq(mask, 1)
            Q_targets = rew

            _, action_probs_next, log_pis_next = self.mac.get_act_probs\
                ([[next_obs[idx],next_feat[idx]], next_avail[idx]])
            Q_target1_next = self.mac.agent.critic1_target([next_obs[idx],next_feat[idx]])
            Q_target2_next = self.mac.agent.critic2_target([next_obs[idx],next_feat[idx]])
            V1_next = (action_probs_next.cuda() * Q_target1_next).sum(1)
            V2_next = (action_probs_next.cuda() * Q_target2_next).sum(1)
            V_target_next = th.min(V1_next,V2_next) - (self.mac.agent.alpha.cuda()* log_pis_next.cuda()).sum(1)
            # Compute Q targets for current states (y_i)
            Q_targets[idx] = rew[idx] + (self.args.gamma * V_target_next)

        # Compute critic loss
        critic1_loss = 0.5 * F.mse_loss(V1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(V2, Q_targets)

        # Update critics
        # critic 1
        self.mac.agent.critic1_optimizer.zero_grad()
        critic1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.mac.agent.critic1.parameters(), self.clip_grad_param)
        self.mac.agent.critic1_optimizer.step()
        
        # critic 2
        self.mac.agent.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.mac.agent.critic2.parameters(), self.clip_grad_param)
        self.mac.agent.critic2_optimizer.step()        

        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.soft_update(self.mac.agent.critic1, self.mac.agent.critic1_target)
            self.soft_update(self.mac.agent.critic2, self.mac.agent.critic2_target)
            self.learn_cnt = 0

        
        return {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item()
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
    