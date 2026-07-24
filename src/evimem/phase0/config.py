from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union


Horizon = Union[int, str]


@dataclass(frozen=True)
class Phase0Config:
    """Composable Phase-0 configuration without model dependencies."""

    carrier: str = "sparse"
    token_cap: str = "B_t"
    horizon: Optional[Horizon] = 32
    resident_slots: int = 2048
    token_budget: Optional[int] = None
    equal_token_quota: Optional[int] = None
    cell_size: float = 0.25
    grid_size: int = 80
    feature_dim: int = 1152
    ring_capacity: int = 33
    seed: int = 0

    def __post_init__(self) -> None:
        if self.carrier not in {"gavln", "sparse"}:
            raise ValueError("carrier must be 'gavln' or 'sparse'")
        if self.token_cap not in {"none", "B_t", "K_eq"}:
            raise ValueError("token_cap must be one of: none, B_t, K_eq")
        if self.carrier == "gavln" and self.horizon is not None:
            raise ValueError("gavln carrier does not take a horizon")
        if self.carrier == "sparse" and self.horizon not in {8, 32, 64, "route"}:
            raise ValueError("sparse horizon must be 8, 32, 64, or 'route'")
        if self.carrier == "gavln" and self.token_cap == "none":
            if self.token_budget is not None or self.equal_token_quota is not None:
                raise ValueError("uncapped gavln cannot set a token budget")
        if self.token_cap == "B_t" and self.token_budget is not None:
            if self.token_budget <= 0:
                raise ValueError("token_budget must be positive")
        if self.token_cap == "K_eq" and self.equal_token_quota is not None:
            if self.equal_token_quota <= 0:
                raise ValueError("equal_token_quota must be positive")
        if self.resident_slots <= 0:
            raise ValueError("resident_slots must be positive")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if self.grid_size <= 0 or self.grid_size % 2:
            raise ValueError("grid_size must be a positive even integer")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.ring_capacity != 33:
            raise ValueError("Phase-0 observation ring capacity is fixed at 33")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    @property
    def experiment_mode(self) -> str:
        if self.carrier == "gavln":
            return "gavln" if self.token_cap == "none" else "gavln_cap"
        return "sparse_h{}".format(self.horizon)

    @property
    def horizon_actions(self) -> Optional[int]:
        if self.horizon == "route" or self.horizon is None:
            return None
        return int(self.horizon)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["experiment_mode"] = self.experiment_mode
        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Phase0Config":
        values = dict(values)
        values.pop("experiment_mode", None)
        return cls(**values)
