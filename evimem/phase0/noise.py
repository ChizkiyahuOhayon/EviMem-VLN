from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .memory import wrap_yaw


GENERATOR_VERSION = "evimem-phase0-noise-v1"
PRODUCTION_FORBIDDEN_KEYS = frozenset(
    {
        "gt_pose",
        "gt_depth",
        "ground_truth_pose",
        "ground_truth_depth",
        "corruption_ancestry",
        "ancestry",
        "draw_id",
        "corruption_draw_id",
        "reference_path",
        "geodesic_progress",
    }
)


@dataclass(frozen=True)
class NoiseTrace:
    translation: np.ndarray
    yaw: np.ndarray
    depth_offset: np.ndarray

    def matrix(self) -> np.ndarray:
        return np.column_stack((self.translation, self.yaw, self.depth_offset)).astype(
            np.float64, copy=False
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for array in (self.translation, self.yaw, self.depth_offset):
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True)
class MatchedNoisePair:
    seed: int
    rho: float
    iid: NoiseTrace
    correlated: NoiseTrace


def _scalar_rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def translation_rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(np.square(array), axis=1))))


def lag1_autocorrelation(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    correlations: List[float] = []
    for column in range(array.shape[1]):
        left = array[:-1, column]
        right = array[1:, column]
        if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def _scale_scalar_to_rms(values: np.ndarray, target_rms: float) -> np.ndarray:
    current = _scalar_rms(values)
    if current == 0:
        raise ValueError("cannot scale a zero trace")
    return np.asarray(values, dtype=np.float64) * (float(target_rms) / current)


def _scale_translation_to_rms(values: np.ndarray, target_rms: float) -> np.ndarray:
    current = translation_rms(values)
    if current == 0:
        raise ValueError("cannot scale a zero translation trace")
    return np.asarray(values, dtype=np.float64) * (float(target_rms) / current)


def generate_matched_noise_pair(
    seed: int,
    length: int = 512,
    rho: float = 0.95,
    translation_target_rms: float = 0.05,
    yaw_target_rms: float = math.radians(5.0),
    depth_target_rms: float = 0.05,
) -> MatchedNoisePair:
    if length < 2 or length > 512:
        raise ValueError("trace length must be in [2, 512]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if min(translation_target_rms, yaw_target_rms, depth_target_rms) <= 0:
        raise ValueError("target RMS values must be positive")

    generator = np.random.default_rng(int(seed))
    innovations = generator.standard_normal((length, 4))
    iid_values = innovations.copy()
    ar_values = np.empty_like(innovations)
    ar_values[0] = innovations[0]
    innovation_scale = math.sqrt(1.0 - rho * rho)
    for step in range(1, length):
        ar_values[step] = rho * ar_values[step - 1] + innovation_scale * innovations[step]

    iid = NoiseTrace(
        translation=_scale_translation_to_rms(
            iid_values[:, :2], translation_target_rms
        ),
        yaw=_scale_scalar_to_rms(iid_values[:, 2], yaw_target_rms),
        depth_offset=_scale_scalar_to_rms(iid_values[:, 3], depth_target_rms),
    )
    correlated = NoiseTrace(
        translation=_scale_translation_to_rms(
            ar_values[:, :2], translation_target_rms
        ),
        yaw=_scale_scalar_to_rms(ar_values[:, 2], yaw_target_rms),
        depth_offset=_scale_scalar_to_rms(ar_values[:, 3], depth_target_rms),
    )
    return MatchedNoisePair(seed=int(seed), rho=float(rho), iid=iid, correlated=correlated)


def time_permute_trace(trace: NoiseTrace, seed: int) -> NoiseTrace:
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(trace.yaw.shape[0])
    return NoiseTrace(
        translation=trace.translation[order].copy(),
        yaw=trace.yaw[order].copy(),
        depth_offset=trace.depth_offset[order].copy(),
    )


def se2_exp(local_twist: Sequence[float]) -> Tuple[np.ndarray, float]:
    twist = np.asarray(local_twist, dtype=np.float64)
    if twist.shape != (3,):
        raise ValueError("local_twist must be [vx, vy, yaw]")
    vx, vy, theta = (float(value) for value in twist)
    if abs(theta) < 1e-8:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0
        b = theta / 2.0 - theta * theta2 / 24.0
    else:
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta
    translation = np.asarray(
        [a * vx - b * vy, b * vx + a * vy], dtype=np.float64
    )
    return translation, theta


def apply_local_se2(
    pose_xy_yaw: Sequence[float], local_twist: Sequence[float]
) -> np.ndarray:
    """Right-compose ``T_gt * Exp(xi)`` in the agent-local frame."""
    pose = np.asarray(pose_xy_yaw, dtype=np.float64)
    if pose.shape != (3,):
        raise ValueError("pose must be [x, y, yaw]")
    local_translation, theta = se2_exp(local_twist)
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    world_translation = np.asarray(
        [
            cosine * local_translation[0] - sine * local_translation[1],
            sine * local_translation[0] + cosine * local_translation[1],
        ]
    )
    return np.asarray(
        [
            pose[0] + world_translation[0],
            pose[1] + world_translation[1],
            wrap_yaw(pose[2] + theta),
        ],
        dtype=np.float64,
    )


def apply_depth_offset(
    depth_meters: np.ndarray,
    offset_meters: float,
    sensor_min: float,
    sensor_max: float,
) -> np.ndarray:
    if not 0 < sensor_min < sensor_max:
        raise ValueError("depth sensor range must be positive and ordered")
    depth = np.asarray(depth_meters, dtype=np.float32)
    return np.clip(depth + np.float32(offset_meters), sensor_min, sensor_max)


def assert_production_payload(payload: Any) -> None:
    """Reject evaluator-only metadata from a production mapping payload."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in PRODUCTION_FORBIDDEN_KEYS:
                raise ValueError("forbidden production field: {}".format(key))
            assert_production_payload(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            assert_production_payload(value)


def _trace_metrics(trace: NoiseTrace) -> Dict[str, float]:
    return {
        "translation_rms": translation_rms(trace.translation),
        "yaw_rms": _scalar_rms(trace.yaw),
        "depth_offset_rms": _scalar_rms(trace.depth_offset),
        "translation_lag1": lag1_autocorrelation(trace.translation),
        "yaw_lag1": lag1_autocorrelation(trace.yaw),
        "depth_lag1": lag1_autocorrelation(trace.depth_offset),
    }


def noise_pair_manifest(pair: MatchedNoisePair) -> Dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": pair.seed,
        "shape": list(pair.iid.matrix().shape),
        "rho": pair.rho,
        "iid": {"sha256": pair.iid.sha256(), **_trace_metrics(pair.iid)},
        "correlated": {
            "sha256": pair.correlated.sha256(),
            **_trace_metrics(pair.correlated),
        },
    }


def write_noise_manifest(path: Path, pairs: Iterable[MatchedNoisePair]) -> None:
    payload = {
        "generator_version": GENERATOR_VERSION,
        "traces": [noise_pair_manifest(pair) for pair in pairs],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
