from .basic_controller import VectorMAC
from .ac_controller import ACMAC
REGISTRY = {}

REGISTRY["vectormac"] = VectorMAC
REGISTRY["ac_mac"] = ACMAC
