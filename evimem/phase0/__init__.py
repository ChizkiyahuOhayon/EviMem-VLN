"""Asset-free EviMem Phase-0 reference implementation."""

from .config import Phase0Config
from .integration import Phase0MemoryState
from .manifest import (
    ResumeMismatchError,
    assert_resume_compatible,
    build_manifest,
    partition_episode_ids,
)
from .memory import (
    FallbackRateError,
    QuotaMonitor,
    ResidualUnderfillError,
    SparseLedger,
    agent_to_world,
    blake2b63,
    enforce_exact_quota,
    incidence_grid_from_raw,
    quantize_world,
    rasterize_ledger,
    stable_select,
    world_to_agent,
)
from .noise import (
    apply_depth_offset,
    apply_local_se2,
    assert_production_payload,
    generate_matched_noise_pair,
)
from .ring import MissingObservationError, ObservationRecord, ObservationRing

__all__ = [
    "FallbackRateError",
    "MissingObservationError",
    "ObservationRecord",
    "ObservationRing",
    "Phase0Config",
    "Phase0MemoryState",
    "QuotaMonitor",
    "ResidualUnderfillError",
    "ResumeMismatchError",
    "SparseLedger",
    "agent_to_world",
    "apply_depth_offset",
    "apply_local_se2",
    "assert_production_payload",
    "assert_resume_compatible",
    "blake2b63",
    "build_manifest",
    "enforce_exact_quota",
    "generate_matched_noise_pair",
    "incidence_grid_from_raw",
    "partition_episode_ids",
    "quantize_world",
    "rasterize_ledger",
    "stable_select",
    "world_to_agent",
]
