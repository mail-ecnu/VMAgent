from .ACAgent import ACAgent
from .QmixAgent import QmixAgent
from .DQNAgent import DQNAgent
REGISTRY = {}

REGISTRY['DQNAgent'] = DQNAgent
REGISTRY['QmixAgent'] = QmixAgent
REGISTRY['ACAgent'] = ACAgent
