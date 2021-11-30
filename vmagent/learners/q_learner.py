import copy
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
import time

class QLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac

        self.params = list(mac.parameters())

        self.learn_cnt= 0

        self.optimiser = Adam(self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_param = list(self.target_mac.parameters())
        self.last_target_update_episode = 0
        self.tau = self.args.tau
        self.gpu_enable = True


    def get_td_error(self, batch):
        obs = batch['obs']
        feat = batch['feat']
        next_obs = batch['next_obs']
        next_feat = batch['next_feat']
        action_list = batch['action']
        reward_list = batch['reward']
        mask_list = 1 - batch['done']
        avail = batch['avail']
        next_avail = batch['next_avail']
        y_list = [0]

        obs = th.FloatTensor(np.array(obs))
        feat = th.FloatTensor(np.array(feat))
        avail = th.FloatTensor(np.array(avail))
        a = th.LongTensor(np.array(action_list))#batch.action))
        rew = th.FloatTensor(np.array(reward_list))##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(np.array(mask_list))#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(np.array(next_obs))
        next_feat = th.FloatTensor(np.array(next_feat))
        next_avail = th.FloatTensor(np.array(next_avail))
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

        # import pdb; pdb.set_trace()
        mac_out, _ = self.mac.forward([[obs, feat], avail])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], None])[0]


        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions]


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array=td_error.float().cpu().data.numpy()
        return td_error_array

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


        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions]


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array = td_error.float().cpu().data.numpy()


        # mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # loss = (weight * masked_td_error ** 2).sum() / mask.sum()
        # import pdb; pdb.set_trace()
        # loss = self.lamb2 * (weight * td_error ** 2).sum() / obs.shape[0] + self.lamb * (ep_error **2).sum()/obs.shape[0]
        # import pdb; pdb.set_trace()
        loss = (td_error ** 2).sum() / obs.shape[0]


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.learn_cnt = 0
        return {
            'critic_loss': loss,
        }



    def _update_targets(self):
        for target_param, local_param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        # self.target_mac.load_state(self.mac)

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

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

class QmixLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac

        self.params = list(mac.parameters())

        self.learn_cnt= 0

        self.optimiser = Adam(self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_param = list(self.target_mac.parameters())
        self.last_target_update_episode = 0
        self.tau = self.args.tau
        self.gpu_enable = True


    def get_td_error(self, batch):
        obs = batch['obs']
        feat = batch['feat']
        next_obs = batch['next_obs']
        next_feat = batch['next_feat']
        action_list = batch['action']
        reward_list = batch['reward']
        mask_list = 1 - batch['done']
        avail = batch['avail']
        next_avail = batch['next_avail']
        y_list = [0]

        obs = th.FloatTensor(np.array(obs))
        feat = th.FloatTensor(np.array(feat))
        avail = th.FloatTensor(np.array(avail))
        a = th.LongTensor(np.array(action_list))#batch.action))
        rew = th.FloatTensor(np.array(reward_list))##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(np.array(mask_list))#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(np.array(next_obs))
        next_feat = th.FloatTensor(np.array(next_feat))
        next_avail = th.FloatTensor(np.array(next_avail))
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

        # import pdb; pdb.set_trace()
        mac_out, _ = self.mac.forward([[obs, feat], avail])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], None])[0]


        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions]


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array=td_error.float().cpu().data.numpy()
        return td_error_array

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
        feat_z = th.zeros_like(feat)
        avail = th.FloatTensor(avail)
        a = th.LongTensor(action_list)#batch.action))
        rew = th.FloatTensor(reward_list)##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(mask_list)#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(next_obs)
        next_feat = th.FloatTensor(next_feat)
        next_feat_z = th.zeros_like(next_feat)
        next_avail = th.FloatTensor(next_avail)
        ind =  th.arange(a.shape[0])
        if self.gpu_enable:
            obs = obs.cuda()
            feat = feat.cuda()
            feat_z = feat_z.cuda()
            avail = avail.cuda()
            a = a.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()
            next_feat = next_feat.cuda()
            next_feat_z = next_feat_z.cuda()
            next_avail = next_avail.cuda()
            ind =  ind.cuda()
        # Calculate estimated Q-Values

        mac_out, _ = self.mac.forward([[obs, feat], avail])
        z_mac_out, _ = self.mac.forward([[obs, feat_z], avail])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  - z_mac_out[ind, a]# Remove the last dim
        chosen_action_qvals += th.sum(z_mac_out, 1) 
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], next_avail])[0]
        z_target_mac_out = self.target_mac.forward([[next_obs, next_feat_z], next_avail])[0]



        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions] - z_target_mac_out[ind, cur_max_actions]
        target_max_qvals += th.sum(z_target_mac_out, 1)


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array = td_error.float().cpu().data.numpy()

        loss = (td_error ** 2).sum() / obs.shape[0]


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.learn_cnt = 0
        return {
            'critic_loss': loss
        }



    def _update_targets(self):
        for target_param, local_param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        # self.target_mac.load_state(self.mac)

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

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()