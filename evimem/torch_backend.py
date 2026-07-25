from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Optional, Tuple

import torch

from evimem.phase0.memory import blake2b63
from gavln.memory.base import MemoryBackendConfig, MemoryBatch, MemoryTokens
from gavln.memory.gavln_backend import _agent_grid, _stable_select


class EviMemTorchBackend:
    """Persistent fixed-capacity sparse world memory used in production inference."""

    def __init__(self, config: MemoryBackendConfig) -> None:
        if config.name != "evimem":
            raise ValueError("EviMemTorchBackend requires name='evimem'")
        self.config = config
        self.reset("uninitialized")

    def reset(self, episode_id: str) -> None:
        self.episode_id = str(episode_id)
        self.last_observation_id = -1
        self.keys: Optional[torch.Tensor] = None
        self.features: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.support_count: Optional[torch.Tensor] = None
        self.first_step: Optional[torch.Tensor] = None
        self.last_step: Optional[torch.Tensor] = None
        self.reservoir_key: Optional[torch.Tensor] = None
        self.active: Optional[torch.Tensor] = None
        self._key_to_slot: Dict[Tuple[int, int], int] = {}
        self._slot_tuple: Dict[int, Tuple[int, int, int]] = {}
        self._free_slots = list(range(self.config.resident_slots))
        heapq.heapify(self._free_slots)
        self.observation_buffer = deque(maxlen=33)

    def _initialize(self, device: torch.device, dtype: torch.dtype) -> None:
        slots = self.config.resident_slots
        dim = self.config.feature_dim
        self.keys = torch.zeros((slots, 2), dtype=torch.int64, device=device)
        self.features = torch.zeros((slots, dim), dtype=dtype, device=device)
        self.weights = torch.zeros(slots, dtype=torch.float32, device=device)
        self.support_count = torch.zeros(slots, dtype=torch.int32, device=device)
        self.first_step = torch.zeros(slots, dtype=torch.int64, device=device)
        self.last_step = torch.zeros(slots, dtype=torch.int64, device=device)
        self.reservoir_key = torch.zeros(slots, dtype=torch.int64, device=device)
        self.active = torch.zeros(slots, dtype=torch.bool, device=device)

    def _require_state(self) -> tuple[torch.Tensor, ...]:
        tensors = (
            self.keys,
            self.features,
            self.weights,
            self.support_count,
            self.first_step,
            self.last_step,
            self.reservoir_key,
            self.active,
        )
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("EviMem state has not been initialized")
        return tensors  # type: ignore[return-value]

    @property
    def active_count(self) -> int:
        return len(self._key_to_slot)

    def _clear_slot(self, slot: int) -> None:
        keys, features, weights, support, first, last, reservoir, active = (
            self._require_state()
        )
        old_key = (int(keys[slot, 0].item()), int(keys[slot, 1].item()))
        self._key_to_slot.pop(old_key, None)
        self._slot_tuple.pop(slot, None)
        keys[slot].zero_()
        features[slot].zero_()
        weights[slot] = 0
        support[slot] = 0
        first[slot] = 0
        last[slot] = 0
        reservoir[slot] = 0
        active[slot] = False

    def _expire(self, action_step: int) -> None:
        if self.config.horizon == "route" or self.active_count == 0:
            return
        *_, last_step, _, active = self._require_state()
        expired = torch.nonzero(
            active & ((int(action_step) - last_step) > int(self.config.horizon)),
            as_tuple=False,
        ).flatten()
        for slot in expired.cpu().tolist():
            self._clear_slot(int(slot))
            heapq.heappush(self._free_slots, int(slot))

    def _write_new(
        self,
        slot: int,
        key: Tuple[int, int],
        feature: torch.Tensor,
        action_step: int,
        candidate_hash: int,
    ) -> None:
        keys, features, weights, support, first, last, reservoir, active = (
            self._require_state()
        )
        keys[slot] = torch.tensor(key, dtype=keys.dtype, device=keys.device)
        features[slot] = feature.to(dtype=features.dtype)
        weights[slot] = 1
        support[slot] = 1
        first[slot] = int(action_step)
        last[slot] = int(action_step)
        reservoir[slot] = int(candidate_hash)
        active[slot] = True
        self._key_to_slot[key] = slot
        self._slot_tuple[slot] = (candidate_hash, key[0], key[1])

    def _update_observation(
        self,
        observation_id: int,
        action_step: int,
        world_xy: torch.Tensor,
        patch_features: torch.Tensor,
    ) -> None:
        if observation_id <= self.last_observation_id:
            return
        self._expire(action_step)
        self.last_observation_id = int(observation_id)

        points = world_xy.reshape(-1, 2)
        values = patch_features.reshape(-1, self.config.feature_dim)
        finite = torch.isfinite(points).all(dim=1) & torch.isfinite(values).all(dim=1)
        points = points[finite]
        values = values[finite]
        if points.numel() == 0:
            return

        quantized = torch.floor(points / self.config.cell_size).to(torch.int64)
        unique_keys, inverse = torch.unique(
            quantized,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        sums = torch.zeros(
            (unique_keys.shape[0], self.config.feature_dim),
            dtype=torch.float32,
            device=values.device,
        )
        sums.index_add_(0, inverse, values.float())
        counts = torch.bincount(inverse, minlength=unique_keys.shape[0]).float()
        means = sums / counts[:, None]

        keys, features, weights, support, _, last, _, _ = self._require_state()
        for index, key_values in enumerate(unique_keys.cpu().tolist()):
            key = (int(key_values[0]), int(key_values[1]))
            feature = means[index]
            slot = self._key_to_slot.get(key)
            if slot is not None:
                old_weight = weights[slot]
                updated = (
                    old_weight * features[slot].float() + feature
                ) / (old_weight + 1.0)
                features[slot] = updated.to(dtype=features.dtype)
                weights[slot] = old_weight + 1.0
                support[slot] = torch.clamp(support[slot] + 1, max=65535)
                last[slot] = int(action_step)
                continue

            candidate_hash = blake2b63(
                self.config.seed,
                self.episode_id,
                key[0],
                key[1],
            )
            candidate_tuple = (candidate_hash, key[0], key[1])
            if self._free_slots:
                slot = heapq.heappop(self._free_slots)
            else:
                worst_slot, worst_tuple = max(
                    self._slot_tuple.items(),
                    key=lambda item: item[1],
                )
                if candidate_tuple >= worst_tuple:
                    continue
                slot = int(worst_slot)
                self._clear_slot(slot)
            self._write_new(slot, key, feature, action_step, candidate_hash)

        if self.active_count > self.config.resident_slots:
            raise AssertionError("EviMem resident capacity exceeded")

    def update(self, batch: MemoryBatch) -> None:
        if self.features is None:
            self._initialize(
                device=batch.siglip_features.device,
                dtype=batch.siglip_features.dtype,
            )
        elif (
            self.features.device != batch.siglip_features.device
            or self.features.dtype != batch.siglip_features.dtype
        ):
            raise RuntimeError("EviMem device/dtype changed within an episode")

        observation_ids = batch.observation_ids.reshape(-1).cpu().tolist()
        action_steps = batch.action_steps.reshape(-1).cpu().tolist()
        for frame_index, (observation_id, action_step) in enumerate(
            zip(observation_ids, action_steps)
        ):
            if int(observation_id) > self.last_observation_id:
                self.observation_buffer.append(
                    {
                        "observation_id": int(observation_id),
                        "action_step": int(action_step),
                        "siglip_features": batch.siglip_features[frame_index].detach(),
                        "siglip_world_xy": batch.siglip_world_xy[frame_index].detach(),
                        "vggt_features": batch.vggt_features[frame_index].detach(),
                        "vggt_world_xy": batch.vggt_world_xy[frame_index].detach(),
                    }
                )
            frame_world = torch.cat(
                (
                    batch.siglip_world_xy[frame_index].reshape(-1, 2),
                    batch.vggt_world_xy[frame_index].reshape(-1, 2),
                ),
                dim=0,
            )
            frame_features = torch.cat(
                (
                    batch.siglip_features[frame_index].reshape(
                        -1, self.config.feature_dim
                    ),
                    batch.vggt_features[frame_index].reshape(
                        -1, self.config.feature_dim
                    ),
                ),
                dim=0,
            )
            self._update_observation(
                int(observation_id),
                int(action_step),
                frame_world,
                frame_features,
            )

    def build_tokens(self, batch: MemoryBatch) -> MemoryTokens:
        if self.features is None:
            raise RuntimeError("EviMem update must run before token readout")
        if (
            self.features.device != batch.siglip_features.device
            or self.features.dtype != batch.siglip_features.dtype
        ):
            raise RuntimeError("EviMem device/dtype changed within an episode")
        keys, features, _, support, _, _, _, active = self._require_state()
        active_slots = torch.nonzero(active, as_tuple=False).flatten()
        cells = self.config.grid_size * self.config.grid_size
        grid_features = torch.zeros(
            (cells, self.config.feature_dim),
            dtype=torch.float32,
            device=features.device,
        )
        incidence = torch.zeros(cells, dtype=torch.float32, device=features.device)
        if active_slots.numel():
            centers = (
                keys[active_slots].to(dtype=features.dtype) + 0.5
            ) * self.config.cell_size
            flat, valid = _agent_grid(
                centers,
                batch.agent_position,
                batch.agent_rotation,
                self.config.cell_size,
                self.config.grid_size,
            )
            slots = active_slots[valid]
            flat = flat[valid]
            weights = support[slots].float()
            grid_features.scatter_add_(
                0,
                flat[:, None].expand(-1, self.config.feature_dim),
                features[slots].float() * weights[:, None],
            )
            incidence.scatter_add_(0, flat, weights)
            nonzero = incidence > 0
            grid_features[nonzero] = (
                grid_features[nonzero]
                / incidence[nonzero, None]
            )
        selected = _stable_select(
            grid_features,
            incidence,
            self.config.token_budget,
        )
        return MemoryTokens(
            features=selected.features,
            flat_indices=selected.flat_indices,
            incidence=selected.incidence,
            resident_count=self.active_count,
        )
