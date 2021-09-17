REGISTRY = {}

from .DQNAgent import DQNAgent
from .QmixAgent import QmixAgent
REGISTRY['DQNAgent'] = DQNAgent
REGISTRY['QmixAgent'] = QmixAgent

