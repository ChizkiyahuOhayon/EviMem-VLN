from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


DTYPE_TOLERANCES = {
    "float32": (1e-6, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1.6e-2),
}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    # Narrow Torch adapter: importing this module never imports torch.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        if "bfloat16" in str(getattr(value, "dtype", "")):
            raise NotImplementedError(
                "Torch BF16 trace capture requires the staged Torch environment "
                "and A0 integration validation"
            )
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_summary(value: Any) -> Dict[str, Any]:
    array = np.ascontiguousarray(_as_numpy(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


class LegacyTraceRecorder:
    """Asset-free recorder for A0 fixtures at the legacy/ring boundary."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.output_dir / "refresh_trace.jsonl"

    def record_refresh(
        self,
        refresh_id: str,
        ids: Any,
        frame_ids: Any,
        gather_order: Any,
        nonzero_mask: Any,
        nonzero_indices: Any,
        raw_float_inputs: Mapping[str, Any],
        computed_tensors: Mapping[str, Any],
        action_logits: Any,
    ) -> Path:
        tensors: Dict[str, np.ndarray] = {
            "ids": _as_numpy(ids),
            "frame_ids": _as_numpy(frame_ids),
            "gather_order": _as_numpy(gather_order),
            "nonzero_mask": _as_numpy(nonzero_mask),
            "nonzero_indices": _as_numpy(nonzero_indices),
            "action_logits": _as_numpy(action_logits),
        }
        tensors.update(
            {"raw__{}".format(name): _as_numpy(value) for name, value in raw_float_inputs.items()}
        )
        tensors.update(
            {
                "computed__{}".format(name): _as_numpy(value)
                for name, value in computed_tensors.items()
            }
        )
        fixture_path = self.output_dir / "refresh_{}.npz".format(refresh_id)
        np.savez_compressed(fixture_path, **tensors)
        record = {
            "refresh_id": str(refresh_id),
            "fixture": fixture_path.name,
            "tensors": {name: tensor_summary(value) for name, value in tensors.items()},
        }
        with self._manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return fixture_path


def compare_a0a_fixture(legacy_path: Path, ring_path: Path) -> None:
    with np.load(legacy_path, allow_pickle=False) as legacy, np.load(
        ring_path, allow_pickle=False
    ) as ring:
        if set(legacy.files) != set(ring.files):
            raise AssertionError("fixture tensor names differ")
        for name in legacy.files:
            expected = legacy[name]
            actual = ring[name]
            if name.startswith("raw__") or expected.dtype.kind in "biu":
                if expected.dtype != actual.dtype or not np.array_equal(expected, actual):
                    raise AssertionError("exact A0a mismatch: {}".format(name))
                continue
            tolerance_key = str(expected.dtype)
            if tolerance_key not in DTYPE_TOLERANCES:
                raise AssertionError("unsupported A0a dtype: {}".format(expected.dtype))
            atol, rtol = DTYPE_TOLERANCES[tolerance_key]
            np.testing.assert_allclose(expected, actual, atol=atol, rtol=rtol)


def action_position_kl(p_gavln: np.ndarray, p_sparse: np.ndarray) -> float:
    left = np.asarray(p_gavln, dtype=np.float64)
    right = np.asarray(p_sparse, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("action probabilities must both be [positions, vocabulary]")
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("KL inputs must be strictly positive probabilities")
    left = left / left.sum(axis=1, keepdims=True)
    right = right / right.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(left * (np.log(left) - np.log(right)), axis=1)))
