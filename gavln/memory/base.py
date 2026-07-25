from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Union, runtime_checkable


Horizon = Union[int, str]


@dataclass(frozen=True)
class MemoryBackendConfig:
    name: str = "gavln"
    horizon: Horizon = 32
    resident_slots: int = 2048
    token_budget: Optional[int] = None
    seed: int = 0
    cell_size: float = 0.25
    grid_size: int = 80
    feature_dim: int = 1152

    def __post_init__(self) -> None:
        if self.name not in {"gavln", "evimem"}:
            raise ValueError("memory backend must be 'gavln' or 'evimem'")
        if self.horizon not in {8, 32, 64, "route"}:
            raise ValueError("memory horizon must be 8, 32, 64, or 'route'")
        if self.resident_slots <= 0:
            raise ValueError("resident_slots must be positive")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token_budget must be positive when set")
        if self.cell_size <= 0 or self.grid_size <= 0 or self.feature_dim <= 0:
            raise ValueError("cell_size, grid_size, and feature_dim must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True)
class MemoryBatch:
    siglip_features: Any
    siglip_world_xy: Any
    vggt_features: Any
    vggt_world_xy: Any
    observation_ids: Any
    action_steps: Any
    agent_position: Any
    agent_rotation: Any


@dataclass(frozen=True)
class MemoryTokens:
    features: Any
    flat_indices: Any
    incidence: Any
    resident_count: int


@runtime_checkable
class MemoryBackend(Protocol):
    config: MemoryBackendConfig

    def reset(self, episode_id: str) -> None: ...

    def update(self, batch: MemoryBatch) -> None: ...

    def build_tokens(self, batch: MemoryBatch) -> MemoryTokens: ...

    @property
    def active_count(self) -> int: ...
