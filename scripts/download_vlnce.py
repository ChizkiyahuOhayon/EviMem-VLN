#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

ARCHIVES = {
    "r2r": {
        "id": "1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm",
        "filename": "R2R_VLNCE_v1-3.zip",
    },
    "rxr": {
        "id": "145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t",
        "filename": "RxR_VLNCE_v0.zip",
    },
}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_r2r(extracted: Path, datasets_root: Path) -> Dict[str, str]:
    destination = datasets_root / "r2r"
    copied: Dict[str, str] = {}
    for split in ("train", "val_seen", "val_unseen", "test"):
        candidates = sorted(extracted.rglob(f"{split}.json.gz"))
        if not candidates:
            if split == "test":
                continue
            raise RuntimeError(f"R2R archive is missing {split}.json.gz")
        target = destination / split / f"{split}.json.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], target)
        copied[split] = str(target)
    return copied


def _normalize_rxr(extracted: Path, datasets_root: Path) -> Dict[str, str]:
    destination = datasets_root / "RxR_VLNCE_v0"
    destination.mkdir(parents=True, exist_ok=True)
    guide_files = sorted(extracted.rglob("*_guide.json.gz"))
    if not guide_files:
        raise RuntimeError("RxR archive contains no *_guide.json.gz files")
    copied: Dict[str, str] = {}
    for source in guide_files:
        split = source.parent.name
        target = destination / split / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[target.relative_to(destination).as_posix()] = str(target)
        gt_source = source.with_name(source.name.replace("_guide.json.gz", "_guide_gt.json.gz"))
        if gt_source.is_file():
            shutil.copy2(gt_source, target.with_name(gt_source.name))
    return copied


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        required=True,
        help="Canonical asset root, preferably on NAS.",
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(ARCHIVES), default=["r2r", "rxr"])
    parser.add_argument("--force-download", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = args.asset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "vln_data"
    downloads = data_root / "_downloads"
    datasets_root = data_root / "datasets"
    downloads.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {"archives": {}}

    for name in args.datasets:
        import gdown

        spec = ARCHIVES[name]
        archive = downloads / str(spec["filename"])
        if args.force_download or not archive.is_file():
            result = gdown.download(id=str(spec["id"]), output=str(archive), quiet=False)
            if result is None or not archive.is_file():
                raise RuntimeError(f"Google Drive download failed for {name}")
        extracted = downloads / f"{name}_extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, extracted)
        normalized = (
            _normalize_r2r(extracted, datasets_root)
            if name == "r2r"
            else _normalize_rxr(extracted, datasets_root)
        )
        manifest["archives"][name] = {
            "google_drive_id": spec["id"],
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "normalized": normalized,
        }

    manifest_path = data_root / "vlnce_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
