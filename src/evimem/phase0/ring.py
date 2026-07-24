from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


class MissingObservationError(KeyError):
    pass


@dataclass(frozen=True)
class ObservationRecord:
    abs_action_step: int
    generation_id: int
    observation_id: int
    valid: bool
    siglip_image: Any
    vggt_image: Any
    world_xy_siglip: Any
    world_xy_vggt: Any
    position: Any
    rotation: Any


class ObservationRing:
    """Fixed 33-record ring for the inclusive interval [t-32, t]."""

    def __init__(self, capacity: int = 33) -> None:
        if capacity != 33:
            raise ValueError("Phase-0 observation ring capacity is fixed at 33")
        self.capacity = capacity
        self._records: List[Optional[ObservationRecord]] = [None] * capacity
        self._next_generation_id = 0
        self._next_observation_id = 0
        self._last_action_step = -1

    def __len__(self) -> int:
        return sum(record is not None and record.valid for record in self._records)

    @property
    def storage_slots(self) -> int:
        return len(self._records)

    @property
    def next_generation_id(self) -> int:
        return self._next_generation_id

    @property
    def next_observation_id(self) -> int:
        return self._next_observation_id

    def push_generation_event(
        self,
        action_queue_length: int,
        abs_action_step: int,
        siglip_image: Any,
        vggt_image: Any,
        world_xy_siglip: Any,
        world_xy_vggt: Any,
        position: Any,
        rotation: Any,
    ) -> Optional[ObservationRecord]:
        """Push only at the upstream ``len(action_seq) == 0`` event."""
        if action_queue_length < 0:
            raise ValueError("action_queue_length cannot be negative")
        if action_queue_length != 0:
            return None
        return self.push(
            abs_action_step=abs_action_step,
            siglip_image=siglip_image,
            vggt_image=vggt_image,
            world_xy_siglip=world_xy_siglip,
            world_xy_vggt=world_xy_vggt,
            position=position,
            rotation=rotation,
        )

    def push(
        self,
        abs_action_step: int,
        siglip_image: Any,
        vggt_image: Any,
        world_xy_siglip: Any,
        world_xy_vggt: Any,
        position: Any,
        rotation: Any,
    ) -> ObservationRecord:
        if abs_action_step < self._last_action_step:
            raise ValueError("abs_action_step must be monotonic within an episode")
        record = ObservationRecord(
            abs_action_step=int(abs_action_step),
            generation_id=self._next_generation_id,
            observation_id=self._next_observation_id,
            valid=True,
            siglip_image=siglip_image,
            vggt_image=vggt_image,
            world_xy_siglip=world_xy_siglip,
            world_xy_vggt=world_xy_vggt,
            position=position,
            rotation=rotation,
        )
        slot = record.observation_id % self.capacity
        self._records[slot] = record
        self._next_generation_id += 1
        self._next_observation_id += 1
        self._last_action_step = int(abs_action_step)
        return record

    def gather(self, horizon: int, current_step: int) -> List[ObservationRecord]:
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        lower = int(current_step) - int(horizon)
        selected = [
            record
            for record in self._records
            if record is not None
            and record.valid
            and lower <= record.abs_action_step <= current_step
        ]
        return sorted(selected, key=lambda record: (record.generation_id, record.observation_id))

    def gather_legacy(self, frame_ids: Sequence[int]) -> List[ObservationRecord]:
        by_observation_id = {
            record.observation_id: record
            for record in self._records
            if record is not None and record.valid
        }
        result = []
        for requested_id in frame_ids:
            observation_id = int(requested_id)
            if observation_id not in by_observation_id:
                raise MissingObservationError(
                    "legacy observation {} is no longer resident".format(observation_id)
                )
            result.append(by_observation_id[observation_id])
        return result

    def dialogue_reset(self) -> None:
        """Dialogue/KV reset intentionally preserves episode memory."""

    def episode_reset(self) -> None:
        self._records[:] = [None] * self.capacity
        self._next_generation_id = 0
        self._next_observation_id = 0
        self._last_action_step = -1
