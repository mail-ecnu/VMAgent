from .ACCritic import ACCritic
from .SoftQCritic import SACCritic

REGISTRY = {}


REGISTRY['ac_critic'] = ACCritic
REGISTRY['soft_q'] = SACCritic
