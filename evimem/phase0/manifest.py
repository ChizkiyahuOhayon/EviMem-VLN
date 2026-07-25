from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = "evimem-phase0-manifest-v1"
RESUME_CRITICAL_FIELDS = (
    "config_hash",
    "source",
    "asset_hashes",
    "checkpoint_hash",
    "dataset_hash",
    "episode_list_hash",
    "evaluator_hash",
    "noise_trace_hash",
)


class ResumeMismatchError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and ".git" not in candidate.parts
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
    )
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def hash_asset(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"status": "missing", "sha256": None}
    if path.is_file():
        return {"status": "present", "kind": "file", "sha256": sha256_file(path)}
    return {"status": "present", "kind": "directory", "sha256": sha256_tree(path)}


def git_source_state(repo_root: Path) -> Dict[str, Any]:
    root_path = Path(repo_root)
    root = str(root_path)
    commit = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", root, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "worktree_hash": sha256_tree(root_path),
    }


def build_manifest(
    experiment_id: str,
    config: Any,
    source_root: Path,
    asset_paths: Mapping[str, Path],
    checkpoint_hash: Optional[str] = None,
    dataset_hash: Optional[str] = None,
    episode_list_hash: Optional[str] = None,
    evaluator_hash: Optional[str] = None,
    noise_trace_hash: Optional[str] = None,
    registry: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config_payload = _jsonable(config)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": str(experiment_id),
        "config": config_payload,
        "config_hash": sha256_json(config_payload),
        "source": git_source_state(source_root),
        "asset_hashes": {
            str(name): hash_asset(path) for name, path in sorted(asset_paths.items())
        },
        "checkpoint_hash": checkpoint_hash or "MISSING",
        "dataset_hash": dataset_hash or "MISSING",
        "episode_list_hash": episode_list_hash or "MISSING",
        "evaluator_hash": evaluator_hash or "MISSING",
        "noise_trace_hash": noise_trace_hash or "MISSING",
        "registry": _jsonable(registry or {}),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(manifest), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_resume_compatible(
    existing: Mapping[str, Any],
    requested: Mapping[str, Any],
    critical_fields: Sequence[str] = RESUME_CRITICAL_FIELDS,
) -> None:
    mismatches = [
        field
        for field in critical_fields
        if _jsonable(existing.get(field)) != _jsonable(requested.get(field))
    ]
    if mismatches:
        raise ResumeMismatchError(
            "resume rejected; critical fields differ: {}".format(", ".join(mismatches))
        )


def partition_episode_ids(
    episode_ids: Sequence[Any], rank: int, world_size: int
) -> List[Any]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    identifiers = list(episode_ids)
    canonical = [canonical_json(identifier) for identifier in identifiers]
    if len(set(canonical)) != len(canonical):
        raise ValueError("episode IDs must be unique before partitioning")
    return identifiers[rank::world_size]
