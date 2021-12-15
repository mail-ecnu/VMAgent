import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
import time
import torch.nn as nn


class PPOMAC:
    def __init__(self, args):
        self.args = args
        server_num, cpu_num, mem_num = args.N, args.cpu, args.mem
        action_space = server_num*2
        obs_space = [(server_num, 2, 2), (1, 2, 2)]
        self._build_agents(obs_space, action_space, args)

        self.action_selector = action_REGISTRY['ppo'](args)

    def mask_invalid(self, agent_outs, avail_actions):
        if avail_actions is not None:
            avail_actions = avail_actions.reshape(agent_outs.shape)
            try:
                idx = th.eq(avail_actions, 0)
                idx_1 = th.eq(avail_actions, 1)
                agent_outs[idx_1] += 0.1
                agent_outs[idx] = 0
                agent_outs.cpu()
            except:
                import pdb
                pdb.set_trace()
        return agent_outs, avail_actions

    def merge_avail_action(self, avail_actions, candi_actions):
        avail = th.zeros_like(avail_actions)
        for i in range(avail.shape[0]):
            if candi_actions[i] == {}:
                avail[i] = avail_actions[i]
            for key in candi_actions[i]:
                avail[i][key] = 1

        return avail

    def select_actions(self, ep_batch, eps=1):
        # Only select actions for the selected batch elements in bs
        start = time.time()
        agent_outputs, avail_actions = self.forward(ep_batch)
        # print('one step', time.time() - start)
        chosen_actions, action_probs, log_action_probabilities, entropy = self.action_selector.select_action(
            agent_outputs, avail_actions)
        try:
            chosen_actions.cpu().numpy()
        except:
            import pdb; pdb.set_trace()
        return chosen_actions.cpu().numpy()
    
    def get_act_probs(self, ep_batch, eps=1):
        start = time.time()
        agent_outputs, avail_actions = self.forward(ep_batch)
        # print('one step', time.time() - start)
        chosen_actions, action_probs, log_action_probabilities, entropy = self.action_selector.select_action(
            agent_outputs, avail_actions)
        try:
            chosen_actions.cpu().numpy()
        except:
            import pdb; pdb.set_trace()
        return chosen_actions.cpu().numpy(), action_probs, log_action_probabilities, entropy

    def forward(self, ep_batch, isDelta=False):
        agent_inputs, avail_actions = self._build_inputs(ep_batch)
        agent_outs = self.agent.actor_local.forward(agent_inputs)
        if isDelta:
            z_agent_inputs, z_avail_actions = self._build_inputs(ep_batch, is_z=True)
            z_agent_outs = self.agent(z_agent_inputs)
            agent_outs -= z_agent_outs
        agent_outs, avail_actions = self.mask_invalid(agent_outs, avail_actions)
        return agent_outs, avail_actions

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, obs_space, action_space, args):
        self.agent = agent_REGISTRY[self.args.agent](
            obs_space, action_space, args).cuda()

    def _build_inputs(self, states, is_z=False):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        obs_inputs = []
        feat_inputs = []
        if type(states) is dict:
            avail_actions = th.Tensor(states['avail']).cuda()
            # import pdb; pdb.set_trace()
            obs = th.Tensor(states['obs']).cuda()
            feat = th.Tensor(states['feat']).cuda()
            if is_z:
                feat = th.zeros_like(feat)
            return [obs, feat], avail_actions
        else:
            if is_z:
                states[1] = th.zeros_like(states[1])
            return states[0], states[1]
