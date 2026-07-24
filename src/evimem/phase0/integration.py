from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .config import Phase0Config
from .memory import (
    RasterizedGrid,
    SelectionResult,
    SparseLedger,
    enforce_exact_quota,
    incidence_grid_from_raw,
    rasterize_ledger,
    stable_select,
)
from .ring import ObservationRecord, ObservationRing
from .trace import LegacyTraceRecorder


UPSTREAM_COMMIT = "cc6086b7081a346695abecf6821f827e2db44a43"
MODEL_INJECTION_POINT = (
    "gavln/model/model_gavln.py::encode_images, after scatter-mean and before "
    "pos_encoding/mm_projector"
)
EVALUATOR_PUSH_POINT = "gavln/gavln_eval.py, inside len(action_seq) == 0"


class Phase0MemoryState:
    """Narrow reference adapter; importing it never imports GA-VLN heavy dependencies.

    TODO(assets): once the pinned checkpoint/environment is staged, call this adapter
    from the two constants above and run A0a before enabling sparse readout. No model,
    prompt, tokenizer, or action path changes are authorized here.
    """

    def __init__(
        self,
        config: Phase0Config,
        episode_id: Any,
        trace_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.ring = ObservationRing(config.ring_capacity)
        self.ledger = SparseLedger(
            episode_id=episode_id,
            seed=config.seed,
            resident_slots=config.resident_slots,
            feature_dim=config.feature_dim,
            cell_size=config.cell_size,
        )
        self.trace_recorder = (
            LegacyTraceRecorder(trace_dir) if trace_dir is not None else None
        )

    def on_generation_event(
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
        return self.ring.push_generation_event(
            action_queue_length=action_queue_length,
            abs_action_step=abs_action_step,
            siglip_image=siglip_image,
            vggt_image=vggt_image,
            world_xy_siglip=world_xy_siglip,
            world_xy_vggt=world_xy_vggt,
            position=position,
            rotation=rotation,
        )

    def ingest_observation_packets(
        self,
        observation_id: int,
        action_step: int,
        world_xy: np.ndarray,
        patch_features: np.ndarray,
    ) -> None:
        self.ledger.update_observation(
            observation_id=observation_id,
            action_step=action_step,
            world_xy=world_xy,
            patch_features=patch_features,
            horizon=self.config.horizon,
        )

    def sparse_readout(
        self,
        agent_position: Sequence[float],
        agent_yaw: float,
        cap: int,
        fallback_flat_indices: Optional[np.ndarray] = None,
        fallback_features: Optional[np.ndarray] = None,
        fallback_last_step: Optional[np.ndarray] = None,
    ) -> SelectionResult:
        grid = rasterize_ledger(
            self.ledger,
            agent_position=agent_position,
            agent_yaw=agent_yaw,
            grid_size=self.config.grid_size,
        )
        selected = stable_select(grid, cap)
        if self.config.token_cap != "K_eq":
            return selected
        if fallback_flat_indices is None or fallback_features is None or fallback_last_step is None:
            if selected.underfilled:
                raise ValueError("K_eq underfill requires the frozen recency fallback pool")
            return selected
        return enforce_exact_quota(
            primary=selected,
            fallback_flat_indices=fallback_flat_indices,
            fallback_features=fallback_features,
            fallback_last_step=fallback_last_step,
            quota=cap,
        )

    def gavln_cap_readout(
        self,
        base_features: np.ndarray,
        observation_ids: np.ndarray,
        world_xy: np.ndarray,
        agent_position: Sequence[float],
        agent_yaw: float,
        cap: int,
    ) -> SelectionResult:
        features = np.asarray(base_features, dtype=np.float32)
        expected_shape = (
            self.config.grid_size,
            self.config.grid_size,
            self.config.feature_dim,
        )
        if features.shape != expected_shape:
            raise ValueError("base_features must have shape {}".format(expected_shape))
        incidence = incidence_grid_from_raw(
            observation_ids=observation_ids,
            world_xy=world_xy,
            agent_position=agent_position,
            agent_yaw=agent_yaw,
            cell_size=self.config.cell_size,
            grid_size=self.config.grid_size,
        )
        return stable_select(
            RasterizedGrid(
                features=features,
                incidence=incidence,
                nonzero=incidence > 0,
            ),
            cap,
        )

    def dialogue_reset(self) -> None:
        self.ring.dialogue_reset()

    def episode_reset(self, episode_id: Any) -> None:
        self.ring.episode_reset()
        self.ledger.reset(episode_id=episode_id)

    def record_legacy_refresh(self, **kwargs: Any) -> Path:
        if self.trace_recorder is None:
            raise RuntimeError("trace_dir was not configured")
        return self.trace_recorder.record_refresh(**kwargs)
