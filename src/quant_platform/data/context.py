"""Optional extra-asset context import contract. Default scope stays BTC/ETH."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ContextAsset:
    pair:str
    source_sha256:str
    role:str='CONTEXT_ONLY'
    def __post_init__(self):
        if not self.pair.endswith('USD') or self.role!='CONTEXT_ONLY' or len(self.source_sha256)!=64:raise ValueError('context source contract')
    @property
    def tradable(self):return False
