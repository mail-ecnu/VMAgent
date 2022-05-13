import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
import time

class ACMAC:
    def __init__(self, args):
        self.args = args
        server_num, cpu_num, mem_num = args.N, args.cpu, args.mem
        action_space = server_num*2
        obs_space = [(server_num, 2, 2), (1, 2, 2)]
        self._build_agents(obs_space, action_space, args)
        self.action_selector = action_REGISTRY['soft_policies'](args)

    def merge_avail_action(self, avail_actions, candi_actions):
        avail = th.zeros_like(avail_actions)
        for i in range(avail.shape[0]):
            if candi_actions[i] == {}:
                avail[i] = avail_actions[i]
            for key in candi_actions[i]:
                avail[i][key] = 1
        return avail

    def select_actions(self, ep_batch, flag ,eps):
        # Only select actions for the selected batch elements in bs
        start = time.time()
        agent_outputs, avail_actions = self.forward(ep_batch)
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, flag)
        try:
            chosen_actions.cpu().numpy()
        except:
            import pdb; pdb.set_trace()
        return  chosen_actions.cpu().numpy()

    def forward(self, ep_batch):
        agent_inputs, avail_actions = self._build_inputs(ep_batch)
        agent_outs = self.agent(agent_inputs)
        # softmax after masked   
        agent_outs[avail_actions == 0.0] = -1e10
        agent_outs = th.nn.functional.softmax(agent_outs,dim=-1)
        return agent_outs, avail_actions

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, x):
        th.save(self.agent.state_dict(), f"{path}/agent_epoch{x}.th")

    def load_models(self, path, x):
        self.agent.load_state_dict(th.load(f"{path}/agent_epoch{x}.th", map_location=lambda storage, loc: storage))

    def _build_agents(self, obs_space, action_space, args):
        self.agent = agent_REGISTRY[self.args.agent](obs_space, action_space, args).cuda()

    def _build_inputs(self, states, is_z=False):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
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


