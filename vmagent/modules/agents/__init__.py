from .ACAgent import ACAgent
from .DQNAgent_modify import DQNAgent_modify
REGISTRY = {}

REGISTRY['ACAgent'] = ACAgent
REGISTRY['DQNAgent_modify'] = DQNAgent_modify