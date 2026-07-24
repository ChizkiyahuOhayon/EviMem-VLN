#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    revision: str
    relative_dir: str
    allow_patterns: Sequence[str]


MODELS: Dict[str, ModelSpec] = {
    "gavln": ModelSpec(
        name="gavln",
        repo_id="jahhao/gavln_official",
        revision="b97f18b48ffc53ab4145d7bbdb517945f3464162",
        relative_dir="checkpoints/gavln_official",
        allow_patterns=(
            "*.json",
            "*.safetensors",
            "added_tokens.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    ),
    "siglip2": ModelSpec(
        name="siglip2",
        repo_id="google/siglip2-so400m-patch14-384",
        revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        relative_dir="model/siglip-so400m-patch14-384",
        allow_patterns=("*.json", "*.model", "model.safetensors"),
    ),
    "vggt": ModelSpec(
        name="vggt",
        repo_id="facebook/VGGT-1B",
        revision="860abec7937da0a4c03c41d3c269c366e82abdf9",
        relative_dir="model/VGGT-1B",
        allow_patterns=("config.json", "model.safetensors"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(directory: Path, spec: ModelSpec) -> Dict[str, object]:
    files = []
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        if path.name == ".evimem-download.json" or ".cache" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "name": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


def download(spec: ModelSpec, asset_root: Path) -> Dict[str, object]:
    from huggingface_hub import snapshot_download

    destination = asset_root / spec.relative_dir
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        allow_patterns=list(spec.allow_patterns),
        resume_download=True,
    )
    payload = _manifest(destination, spec)
    marker = destination / ".evimem-download.json"
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="Canonical asset root, preferably on NAS; GA-VLN-style subdirectories are created.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(MODELS),
        default=sorted(MODELS),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = args.asset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        payload = download(MODELS[name], root)
        gib = int(payload["total_bytes"]) / (1024 ** 3)
        print(f"{name}: {len(payload['files'])} files, {gib:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
