from __future__ import annotations

from typing import Optional

import torch

from .base import MemoryBackendConfig, MemoryBatch, MemoryTokens


def _agent_grid(
    world_xy: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    cell_size: float,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = world_xy.reshape(-1, 2)
    finite = torch.isfinite(points).all(dim=1)
    relative = points - position.to(device=points.device, dtype=points.dtype)
    local = torch.matmul(
        relative,
        -rotation.to(device=points.device, dtype=points.dtype),
    )
    indices = torch.floor(local / cell_size).to(torch.long) + grid_size // 2
    valid = (
        finite
        & (indices[:, 0] >= 0)
        & (indices[:, 0] < grid_size)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < grid_size)
    )
    flat = indices[:, 1] * grid_size + indices[:, 0]
    return flat, valid


def _stable_select(
    grid_features: torch.Tensor,
    incidence: torch.Tensor,
    token_budget: Optional[int],
) -> MemoryTokens:
    candidates = torch.nonzero(incidence > 0, as_tuple=False).flatten()
    if token_budget is not None and candidates.numel() > token_budget:
        order = torch.argsort(
            incidence[candidates],
            descending=True,
            stable=True,
        )
        candidates = torch.sort(candidates[order[:token_budget]]).values
    return MemoryTokens(
        features=grid_features[candidates].contiguous(),
        flat_indices=candidates,
        incidence=incidence[candidates],
        resident_count=int(candidates.numel()),
    )


class GAVLNMemoryBackend:
    """Original windowed GA-VLN scatter-mean BEV implementation."""

    def __init__(self, config: MemoryBackendConfig) -> None:
        if config.name != "gavln":
            raise ValueError("GAVLNMemoryBackend requires name='gavln'")
        self.config = config

    def reset(self, episode_id: str) -> None:
        del episode_id

    def update(self, batch: MemoryBatch) -> None:
        del batch

    @property
    def active_count(self) -> int:
        return 0

    def build_tokens(self, batch: MemoryBatch) -> MemoryTokens:
        siglip = batch.siglip_features.reshape(-1, self.config.feature_dim)
        vggt = batch.vggt_features.reshape(-1, self.config.feature_dim)
        features = torch.cat((siglip, vggt), dim=0)

        siglip_flat, siglip_valid = _agent_grid(
            batch.siglip_world_xy,
            batch.agent_position,
            batch.agent_rotation,
            self.config.cell_size,
            self.config.grid_size,
        )
        vggt_flat, vggt_valid = _agent_grid(
            batch.vggt_world_xy,
            batch.agent_position,
            batch.agent_rotation,
            self.config.cell_size,
            self.config.grid_size,
        )
        flat = torch.cat((siglip_flat[siglip_valid], vggt_flat[vggt_valid]))
        values = torch.cat((siglip[siglip_valid], vggt[vggt_valid]))

        cells = self.config.grid_size * self.config.grid_size
        sums = torch.zeros(
            (cells, self.config.feature_dim),
            dtype=values.dtype,
            device=values.device,
        )
        counts = torch.zeros(cells, dtype=values.dtype, device=values.device)
        if flat.numel():
            sums.scatter_add_(
                0,
                flat[:, None].expand(-1, self.config.feature_dim),
                values,
            )
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
        nonzero = counts > 0
        sums[nonzero] = sums[nonzero] / counts[nonzero, None]
        return _stable_select(sums, counts, self.config.token_budget)
