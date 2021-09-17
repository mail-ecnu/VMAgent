from .q_learner import QLearner
from .q_learner import QmixLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qmix_learner"] = QmixLearner


