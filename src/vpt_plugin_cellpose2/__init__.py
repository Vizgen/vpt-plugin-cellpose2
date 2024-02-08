from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CellposeSegProperties:
    model_dimensions: str
    channel_map: dict
    model: Optional[str] = None
    custom_weights: Optional[str] = None


@dataclass(frozen=True)
class CellposeSegParameters:
    nuclear_channel: str
    entity_fill_channel: str
    diameter: int
    flow_threshold: float
    cellprob_threshold: float
    minimum_mask_size: int
