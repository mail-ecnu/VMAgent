import torch as th
from torch.distributions import Categorical
import numpy as np

logpath = './mylogs/action_selects.csv'
import csv

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


class NormalACActionSelector():
    def __init__(self, args):
        self.args = args


    def select_action(self, agent_outputs, avail_actions, flag=False):
        action_probs = agent_outputs * avail_actions
        dist = Categorical(action_probs)
        # print('nan values:', th.sum(th.isnan(action_probs)).item())
        if flag == False:
            actions = dist.sample()
        else:
            actions = th.argmax(action_probs,dim=-1)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = th.log(action_probs + z)

        return actions.detach().cpu(), action_probs, log_action_probabilities.detach()


class PPOActionSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_outputs, avail_actions,flag=False):
        action_probs = agent_outputs * avail_actions

        dist = Categorical(action_probs)
        if flag == False:
            actions = dist.sample()
        if flag == True:
            actions = th.argmax(action_probs,dim=-1)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = th.log(action_probs + z)
        dist_entropy = dist.entropy()
        return actions.detach().cpu(), action_probs, log_action_probabilities, dist_entropy


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
REGISTRY["AC"] = NormalACActionSelector
REGISTRY["ppo"] = PPOActionSelector
