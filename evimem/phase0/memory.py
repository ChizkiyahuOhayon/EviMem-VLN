from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


UINT16_MAX = int(np.iinfo(np.uint16).max)
INT32_MIN = int(np.iinfo(np.int32).min)
INT32_MAX = int(np.iinfo(np.int32).max)


class ResidualUnderfillError(RuntimeError):
    pass


class FallbackRateError(RuntimeError):
    pass


def float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round FP32 to raw BF16 bits using round-to-nearest-even."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_to_float32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def quantize_world(world_xy: np.ndarray, cell_size: float = 0.25) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("world_xy must have shape [N, 2]")
    keys64 = np.floor(points / float(cell_size)).astype(np.int64)
    if np.any(keys64 < INT32_MIN) or np.any(keys64 > INT32_MAX):
        raise OverflowError("quantized world key exceeds int32")
    return keys64.astype(np.int32)


def world_cell_centers(keys: np.ndarray, cell_size: float = 0.25) -> np.ndarray:
    keys_array = np.asarray(keys, dtype=np.int32)
    return (keys_array.astype(np.float64) + 0.5) * float(cell_size)


def wrap_yaw(yaw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    wrapped = (np.asarray(yaw) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(yaw) == 0:
        return float(wrapped)
    return wrapped


def world_to_agent(
    world_xy: np.ndarray,
    agent_position: Sequence[float],
    agent_yaw: float,
) -> np.ndarray:
    points = np.asarray(world_xy, dtype=np.float64)
    position = np.asarray(agent_position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or position.shape != (2,):
        raise ValueError("world_xy must be [N,2] and agent_position must be [2]")
    relative = points - position
    cosine = math.cos(float(agent_yaw))
    sine = math.sin(float(agent_yaw))
    x_agent = cosine * relative[:, 0] + sine * relative[:, 1]
    y_agent = -sine * relative[:, 0] + cosine * relative[:, 1]
    return np.stack((x_agent, y_agent), axis=1)


def agent_to_world(
    agent_xy: np.ndarray,
    agent_position: Sequence[float],
    agent_yaw: float,
) -> np.ndarray:
    points = np.asarray(agent_xy, dtype=np.float64)
    position = np.asarray(agent_position, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or position.shape != (2,):
        raise ValueError("agent_xy must be [N,2] and agent_position must be [2]")
    cosine = math.cos(float(agent_yaw))
    sine = math.sin(float(agent_yaw))
    x_world = cosine * points[:, 0] - sine * points[:, 1] + position[0]
    y_world = sine * points[:, 0] + cosine * points[:, 1] + position[1]
    return np.stack((x_world, y_world), axis=1)


def agent_grid_indices(
    agent_xy: np.ndarray,
    cell_size: float = 0.25,
    grid_size: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(agent_xy, dtype=np.float64)
    indices = np.floor(points / float(cell_size)).astype(np.int64) + grid_size // 2
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < grid_size)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < grid_size)
    )
    flat = indices[:, 1] * grid_size + indices[:, 0]
    return indices, flat, valid


def _length_prefix(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, byteorder="little", signed=False) + payload


def blake2b63(
    seed: int,
    episode_id: Any,
    world_key_x: int,
    world_key_y: int,
) -> int:
    """Contract serialization for the uniform bottom-k priority."""
    if seed < 0 or seed > (1 << 64) - 1:
        raise ValueError("seed must fit unsigned 64-bit")
    for coordinate in (world_key_x, world_key_y):
        if coordinate < INT32_MIN or coordinate > INT32_MAX:
            raise ValueError("world key must fit signed 32-bit")
    fields = (
        int(seed).to_bytes(8, byteorder="little", signed=False),
        str(episode_id).encode("utf-8"),
        int(world_key_x).to_bytes(4, byteorder="little", signed=True),
        int(world_key_y).to_bytes(4, byteorder="little", signed=True),
    )
    digest = hashlib.blake2b(
        b"".join(_length_prefix(field) for field in fields), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


@dataclass
class LedgerStats:
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    replaced: int = 0
    expired: int = 0


class SparseLedger:
    """Fixed-shape unique-world-cell ledger with deterministic bottom-k admission."""

    def __init__(
        self,
        episode_id: Any,
        seed: int = 0,
        resident_slots: int = 2048,
        feature_dim: int = 1152,
        cell_size: float = 0.25,
    ) -> None:
        if resident_slots <= 0 or feature_dim <= 0 or cell_size <= 0:
            raise ValueError("ledger dimensions and cell_size must be positive")
        self.episode_id = str(episode_id)
        self.seed = int(seed)
        self.resident_slots = int(resident_slots)
        self.feature_dim = int(feature_dim)
        self.cell_size = float(cell_size)

        self.keys = np.zeros((resident_slots, 2), dtype=np.int32)
        # NumPy has no portable BF16 dtype; uint16 stores the exact BF16 bit pattern.
        self.features = np.zeros((resident_slots, feature_dim), dtype=np.uint16)
        self.weight = np.zeros(resident_slots, dtype=np.float32)
        self.confidence = np.zeros(resident_slots, dtype=np.float32)
        self.priority = np.zeros(resident_slots, dtype=np.float32)
        self.reservoir_key = np.zeros(resident_slots, dtype=np.int64)
        self.first_step = np.zeros(resident_slots, dtype=np.int32)
        self.last_step = np.zeros(resident_slots, dtype=np.int32)
        self.support_count = np.zeros(resident_slots, dtype=np.uint16)
        self.active = np.zeros(resident_slots, dtype=np.bool_)

        self.stats = LedgerStats()
        self._last_observation_id = -1
        self._worst_slot: Optional[int] = None

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active))

    @property
    def resident_overflow(self) -> int:
        return max(0, self.active_count - self.resident_slots)

    @property
    def logical_bytes(self) -> int:
        arrays = (
            self.keys,
            self.features,
            self.weight,
            self.confidence,
            self.priority,
            self.reservoir_key,
            self.first_step,
            self.last_step,
            self.support_count,
            self.active,
        )
        return int(sum(array.nbytes for array in arrays))

    def tensor_schema(self) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        return {
            "keys": ("int32", self.keys.shape),
            "features": ("bfloat16", self.features.shape),
            "weight": ("float32", self.weight.shape),
            "confidence": ("float32", self.confidence.shape),
            "priority": ("float32", self.priority.shape),
            "reservoir_key": ("int64", self.reservoir_key.shape),
            "first_step": ("int32", self.first_step.shape),
            "last_step": ("int32", self.last_step.shape),
            "support_count": ("uint16", self.support_count.shape),
            "active": ("bool", self.active.shape),
        }

    def reset(self, episode_id: Optional[Any] = None) -> None:
        if episode_id is not None:
            self.episode_id = str(episode_id)
        for array in (
            self.keys,
            self.features,
            self.weight,
            self.confidence,
            self.priority,
            self.reservoir_key,
            self.first_step,
            self.last_step,
            self.support_count,
            self.active,
        ):
            array.fill(0)
        self.stats = LedgerStats()
        self._last_observation_id = -1
        self._worst_slot = None

    def _tuple_for_slot(self, slot: int) -> Tuple[int, int, int]:
        return (
            int(self.reservoir_key[slot]),
            int(self.keys[slot, 0]),
            int(self.keys[slot, 1]),
        )

    def _recompute_worst_slot(self) -> None:
        slots = np.flatnonzero(self.active)
        if slots.size == 0:
            self._worst_slot = None
            return
        self._worst_slot = max(
            (int(slot) for slot in slots), key=lambda slot: self._tuple_for_slot(slot)
        )

    def _find_slot(self, key: np.ndarray) -> Optional[int]:
        matches = np.flatnonzero(self.active & np.all(self.keys == key, axis=1))
        if matches.size > 1:
            raise AssertionError("unique-world-cell invariant violated")
        return None if matches.size == 0 else int(matches[0])

    def _clear_slots(self, slots: np.ndarray) -> None:
        if slots.size == 0:
            return
        self.keys[slots] = 0
        self.features[slots] = 0
        self.weight[slots] = 0
        self.confidence[slots] = 0
        self.priority[slots] = 0
        self.reservoir_key[slots] = 0
        self.first_step[slots] = 0
        self.last_step[slots] = 0
        self.support_count[slots] = 0
        self.active[slots] = False

    def expire(self, current_step: int, horizon: Optional[Union[int, str]]) -> int:
        if horizon is None or horizon == "route":
            return 0
        horizon_value = int(horizon)
        if horizon_value < 0:
            raise ValueError("horizon must be nonnegative or 'route'")
        slots = np.flatnonzero(
            self.active & ((int(current_step) - self.last_step.astype(np.int64)) > horizon_value)
        )
        expired_worst = self._worst_slot is not None and self._worst_slot in set(slots.tolist())
        self._clear_slots(slots)
        count = int(slots.size)
        self.stats.expired += count
        if expired_worst:
            self._recompute_worst_slot()
        return count

    def _write_new(
        self,
        slot: int,
        key: np.ndarray,
        feature: np.ndarray,
        action_step: int,
        reservoir_key: int,
    ) -> None:
        self.keys[slot] = key
        self.features[slot] = float32_to_bf16(feature)
        self.weight[slot] = np.float32(1.0)
        self.confidence[slot] = np.float32(0.0)
        self.priority[slot] = np.float32(0.0)
        self.reservoir_key[slot] = np.int64(reservoir_key)
        self.first_step[slot] = np.int32(action_step)
        self.last_step[slot] = np.int32(action_step)
        self.support_count[slot] = np.uint16(1)
        self.active[slot] = True

    def _update_existing(self, slot: int, feature: np.ndarray, action_step: int) -> None:
        old_feature = bf16_to_float32(self.features[slot])
        old_weight = np.float32(self.weight[slot])
        numerator = old_weight * old_feature + np.asarray(feature, dtype=np.float32)
        updated = numerator / np.float32(old_weight + np.float32(1.0))
        self.features[slot] = float32_to_bf16(updated)
        self.weight[slot] = np.float32(old_weight + np.float32(1.0))
        support = min(UINT16_MAX, int(self.support_count[slot]) + 1)
        self.support_count[slot] = np.uint16(support)
        self.last_step[slot] = np.int32(action_step)
        self.stats.updated += 1

    def update_observation(
        self,
        observation_id: int,
        action_step: int,
        world_xy: np.ndarray,
        patch_features: np.ndarray,
        horizon: Optional[Union[int, str]],
    ) -> None:
        if int(observation_id) <= self._last_observation_id:
            raise ValueError("update_observation must be called once per increasing observation_id")
        points = np.asarray(world_xy, dtype=np.float64)
        features = np.asarray(patch_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("patch_features must have shape [N, feature_dim]")
        if points.shape != (features.shape[0], 2):
            raise ValueError("world_xy and patch_features must have the same first dimension")

        # Contract order: expire first, then process this observation's candidates.
        self.expire(current_step=int(action_step), horizon=horizon)
        self._last_observation_id = int(observation_id)
        if features.shape[0] == 0:
            return

        patch_keys = quantize_world(points, self.cell_size)
        unique_keys, inverse = np.unique(patch_keys, axis=0, return_inverse=True)
        observation_features = np.empty(
            (unique_keys.shape[0], self.feature_dim), dtype=np.float32
        )
        for index in range(unique_keys.shape[0]):
            observation_features[index] = features[inverse == index].mean(axis=0, dtype=np.float32)

        for key, feature in zip(unique_keys, observation_features):
            existing_slot = self._find_slot(key)
            if existing_slot is not None:
                self._update_existing(existing_slot, feature, int(action_step))
                continue

            candidate_key = blake2b63(
                self.seed, self.episode_id, int(key[0]), int(key[1])
            )
            candidate_tuple = (candidate_key, int(key[0]), int(key[1]))
            free_slots = np.flatnonzero(~self.active)
            if free_slots.size:
                slot = int(free_slots[0])
                self._write_new(slot, key, feature, int(action_step), candidate_key)
                self.stats.inserted += 1
                if self._worst_slot is None or candidate_tuple > self._tuple_for_slot(
                    self._worst_slot
                ):
                    self._worst_slot = slot
                continue

            if self._worst_slot is None:
                self._recompute_worst_slot()
            assert self._worst_slot is not None
            if candidate_tuple < self._tuple_for_slot(self._worst_slot):
                slot = self._worst_slot
                self._write_new(slot, key, feature, int(action_step), candidate_key)
                self.stats.replaced += 1
                self._recompute_worst_slot()
            else:
                self.stats.rejected += 1

            if self.active_count > self.resident_slots:
                raise AssertionError("resident overflow")

    def active_features_float32(self) -> np.ndarray:
        return bf16_to_float32(self.features[self.active])

    def resident_tuples(self) -> List[Tuple[int, int, int]]:
        return sorted(
            self._tuple_for_slot(int(slot)) for slot in np.flatnonzero(self.active)
        )


@dataclass(frozen=True)
class RasterizedGrid:
    features: np.ndarray
    incidence: np.ndarray
    nonzero: np.ndarray


def scatter_mean(
    flat_indices: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    num_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(flat_indices, dtype=np.int64)
    values = np.asarray(features, dtype=np.float32)
    cell_weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or flat.shape != (values.shape[0],):
        raise ValueError("flat_indices and features have incompatible shapes")
    if cell_weights.shape != flat.shape:
        raise ValueError("weights must match flat_indices")
    if np.any(flat < 0) or np.any(flat >= num_cells):
        raise ValueError("flat index outside grid")
    sums = np.zeros((num_cells, values.shape[1]), dtype=np.float64)
    totals = np.zeros(num_cells, dtype=np.float64)
    np.add.at(sums, flat, values.astype(np.float64) * cell_weights[:, None])
    np.add.at(totals, flat, cell_weights)
    output = np.zeros_like(sums, dtype=np.float32)
    nonzero = totals > 0
    output[nonzero] = (sums[nonzero] / totals[nonzero, None]).astype(np.float32)
    return output, totals


def rasterize_ledger(
    ledger: SparseLedger,
    agent_position: Sequence[float],
    agent_yaw: float,
    grid_size: int = 80,
) -> RasterizedGrid:
    slots = np.flatnonzero(ledger.active)
    feature_dim = ledger.feature_dim
    if slots.size == 0:
        return RasterizedGrid(
            features=np.zeros((grid_size, grid_size, feature_dim), dtype=np.float32),
            incidence=np.zeros((grid_size, grid_size), dtype=np.int64),
            nonzero=np.zeros((grid_size, grid_size), dtype=np.bool_),
        )
    centers = world_cell_centers(ledger.keys[slots], ledger.cell_size)
    local = world_to_agent(centers, agent_position, agent_yaw)
    _, flat, valid = agent_grid_indices(local, ledger.cell_size, grid_size)
    valid_slots = slots[valid]
    valid_flat = flat[valid]
    support = ledger.support_count[valid_slots].astype(np.int64)
    features = bf16_to_float32(ledger.features[valid_slots])
    flat_features, totals = scatter_mean(
        valid_flat, features, support, grid_size * grid_size
    )
    incidence = totals.astype(np.int64)
    return RasterizedGrid(
        features=flat_features.reshape(grid_size, grid_size, feature_dim),
        incidence=incidence.reshape(grid_size, grid_size),
        nonzero=(incidence > 0).reshape(grid_size, grid_size),
    )


def incidence_grid_from_raw(
    observation_ids: np.ndarray,
    world_xy: np.ndarray,
    agent_position: Sequence[float],
    agent_yaw: float,
    cell_size: float = 0.25,
    grid_size: int = 80,
) -> np.ndarray:
    observation_ids = np.asarray(observation_ids, dtype=np.int64)
    world_keys = quantize_world(world_xy, cell_size)
    if observation_ids.shape != (world_keys.shape[0],):
        raise ValueError("observation_ids must match world_xy")
    triples = np.column_stack((observation_ids, world_keys.astype(np.int64)))
    unique_triples = np.unique(triples, axis=0)
    unique_world_keys, counts = np.unique(
        unique_triples[:, 1:3].astype(np.int32), axis=0, return_counts=True
    )
    centers = world_cell_centers(unique_world_keys, cell_size)
    local = world_to_agent(centers, agent_position, agent_yaw)
    _, flat, valid = agent_grid_indices(local, cell_size, grid_size)
    output = np.zeros(grid_size * grid_size, dtype=np.int64)
    np.add.at(output, flat[valid], counts[valid].astype(np.int64))
    return output.reshape(grid_size, grid_size)


@dataclass(frozen=True)
class SelectionResult:
    flat_indices: np.ndarray
    features: np.ndarray
    incidence: np.ndarray
    requested: int
    underfilled: bool
    fallback_count: int = 0


def stable_select(grid: RasterizedGrid, cap: int) -> SelectionResult:
    if cap < 0:
        raise ValueError("cap must be nonnegative")
    flat_incidence = np.asarray(grid.incidence).reshape(-1)
    flat_features = np.asarray(grid.features).reshape(flat_incidence.size, -1)
    candidates = np.flatnonzero(flat_incidence > 0)
    if candidates.size:
        rank_order = np.lexsort((candidates, -flat_incidence[candidates]))
        selected = np.sort(candidates[rank_order[: min(cap, candidates.size)]])
    else:
        selected = np.empty(0, dtype=np.int64)
    return SelectionResult(
        flat_indices=selected.astype(np.int64, copy=False),
        features=flat_features[selected].astype(np.float32, copy=True),
        incidence=flat_incidence[selected].astype(np.int64, copy=True),
        requested=int(cap),
        underfilled=selected.size < cap,
        fallback_count=0,
    )


def enforce_exact_quota(
    primary: SelectionResult,
    fallback_flat_indices: np.ndarray,
    fallback_features: np.ndarray,
    fallback_last_step: np.ndarray,
    quota: int,
    fallback_incidence: Optional[np.ndarray] = None,
) -> SelectionResult:
    if quota < 0:
        raise ValueError("quota must be nonnegative")
    if primary.flat_indices.size > quota:
        raise ValueError("primary selection exceeds quota")
    fallback_indices = np.asarray(fallback_flat_indices, dtype=np.int64)
    fallback_values = np.asarray(fallback_features, dtype=np.float32)
    recency = np.asarray(fallback_last_step, dtype=np.int64)
    if fallback_values.ndim != 2 or fallback_values.shape[0] != fallback_indices.size:
        raise ValueError("fallback features and indices have incompatible shapes")
    if recency.shape != fallback_indices.shape:
        raise ValueError("fallback_last_step must match fallback indices")
    if fallback_incidence is None:
        fallback_counts = np.ones(fallback_indices.size, dtype=np.int64)
    else:
        fallback_counts = np.asarray(fallback_incidence, dtype=np.int64)
        if fallback_counts.shape != fallback_indices.shape:
            raise ValueError("fallback_incidence must match fallback indices")

    feature_by_index = {
        int(index): np.asarray(feature, dtype=np.float32).copy()
        for index, feature in zip(primary.flat_indices, primary.features)
    }
    incidence_by_index = {
        int(index): int(count)
        for index, count in zip(primary.flat_indices, primary.incidence)
    }
    fallback_added = 0
    fallback_order = np.lexsort((fallback_indices, -recency))
    for position in fallback_order:
        flat_index = int(fallback_indices[position])
        if flat_index in feature_by_index:
            continue
        feature_by_index[flat_index] = fallback_values[position].copy()
        incidence_by_index[flat_index] = int(fallback_counts[position])
        fallback_added += 1
        if len(feature_by_index) == quota:
            break

    if len(feature_by_index) != quota:
        raise ResidualUnderfillError(
            "exact quota {} cannot be filled with unique active cells".format(quota)
        )
    selected = np.asarray(sorted(feature_by_index), dtype=np.int64)
    return SelectionResult(
        flat_indices=selected,
        features=np.stack([feature_by_index[int(index)] for index in selected]),
        incidence=np.asarray(
            [incidence_by_index[int(index)] for index in selected], dtype=np.int64
        ),
        requested=quota,
        underfilled=False,
        fallback_count=fallback_added,
    )


class QuotaMonitor:
    def __init__(self, max_fallback_rate: float = 0.05) -> None:
        self.max_fallback_rate = float(max_fallback_rate)
        self.refreshes = 0
        self.fallback_refreshes = 0

    @property
    def fallback_rate(self) -> float:
        if self.refreshes == 0:
            return 0.0
        return self.fallback_refreshes / self.refreshes

    def observe(self, result: SelectionResult) -> None:
        if result.underfilled or result.flat_indices.size != result.requested:
            raise ResidualUnderfillError("quota result is underfilled")
        self.refreshes += 1
        if result.fallback_count:
            self.fallback_refreshes += 1

    def validate(self, locked_test: bool) -> None:
        if self.fallback_rate > self.max_fallback_rate:
            phase = "locked test" if locked_test else "calibration"
            raise FallbackRateError(
                "{} fallback rate {:.3f} exceeds {:.3f}".format(
                    phase, self.fallback_rate, self.max_fallback_rate
                )
            )
