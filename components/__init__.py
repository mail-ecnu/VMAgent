REGISTRY = {}

from .replay_memory import ReplayMemory

REGISTRY["replay"] = ReplayMemory
