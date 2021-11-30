from .NormalACAgent import NormalACAgent
from .SACAgent import SACAgent
from .QmixAgent import QmixAgent
from .DQNAgent import DQNAgent
REGISTRY = {}


REGISTRY['SACAgent'] = SACAgent
REGISTRY['DQNAgent'] = DQNAgent
REGISTRY['QmixAgent'] = QmixAgent
REGISTRY['NormalACAgent'] = NormalACAgent
