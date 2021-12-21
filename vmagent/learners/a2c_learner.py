import copy
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class A2CLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac
        self.params = list(mac.parameters())
        self.learn_cnt= 0
        self.optimiser = Adam(self.params, lr=args.lr)
        self.clip_grad_param = 1
        self.gamma = args.gamma

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        print(mac)
        self.target_mac = copy.deepcopy(mac)
        self.target_param = list(self.target_mac.parameters())
        self.last_target_update_episode = 0
        self.tau = self.args.tau
        self.gpu_enable = True
    
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
        # mac_out, _ = self.mac.forward([[obs, feat], avail])
        # Pick the Q-Values for the actions taken by each agent
        # chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        # target_mac_out = self.target_mac.forward([[next_obs, next_feat], next_avail])[0]

        # ---------------------------- update actor ---------------------------- #
        _, action_probs, log_pis = self.mac.get_act_probs([[obs, feat], avail])

        q_values = self.mac.agent.critic([obs, feat])
        # print(log_pis.grad_fn)
        # print(q_values.grad_fn)
        
        # advantages = q_values - V.unsqueeze(-1)

        actor_loss = -(log_pis * q_values).sum(1).mean()

        # print(actor_loss)
        # print(self.mac.agent.actor_local([obs, feat]))
        self.mac.agent.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.mac.agent.actor_optimizer.step()
        # print(self.mac.agent.actor_local([obs, feat]))

        V = (action_probs.cuda() * q_values.cuda()).sum(1)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with th.no_grad():
            idx = th.eq(mask, 1)
            Q_targets = rew
            _, action_probs_next, _ = self.mac.get_act_probs(
                [[next_obs[idx], next_feat[idx]], next_avail[idx]])
            q_next = self.mac.agent.critic([next_obs[idx],next_feat[idx]])

            V_next = (action_probs_next.cuda() * q_next.cuda()).sum(1)
            # Compute Q targets for current states (y_i)
            Q_targets[idx] = rew[idx] + (self.args.gamma *  V_next)

        # Compute critic loss        
        critic_loss = 0.5 * F.mse_loss(V, Q_targets)

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
    