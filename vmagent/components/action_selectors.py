import torch as th
from torch.distributions import Categorical
import numpy as np

REGISTRY = {}

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, eps, avail_actions, eps2):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = eps

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = - \
            float("inf")  # should never be selected!
        avail_actions = avail_actions.float()
        # for q in avail_actions[0]:
        #     if q < 0:
        random_numbers = th.rand_like(agent_inputs[:, 0])
        pick_random = (random_numbers < self.epsilon * (1-eps2)).long()
        pick_default = (random_numbers >= 1 - self.epsilon * eps2).long()
        random_actions = Categorical(avail_actions.float()).sample().long()
        default_actions = -th.ones_like(random_actions)
        picked_actions = pick_random * random_actions + pick_default*default_actions + \
            (1 - pick_random - pick_default) * masked_q_values.max(dim=1)[1]
        # picked_actions = random_actions
        return picked_actions


class SoftPoliciesSelector():


    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, flag=False):  
        if flag:
            picked_actions = th.argmax(agent_inputs,dim=-1)
        else:
            picked_actions = Categorical(agent_inputs).sample()
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["soft_policies"] = SoftPoliciesSelector
