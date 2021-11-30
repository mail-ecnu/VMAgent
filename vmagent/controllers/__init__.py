from .ppo_controller import PPOMAC
from .sac_controller import SACMAC
from .basic_controller import VectorMAC
REGISTRY = {}


REGISTRY["vectormac"] = VectorMAC
REGISTRY["sacmac"] = SACMAC
REGISTRY["ppomac"] = PPOMAC
