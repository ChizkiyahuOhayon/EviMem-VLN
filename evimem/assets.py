from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class AssetCheck:
    name: str
    status: str
    path: str
    detail: str


def _files(path: Path, patterns: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for pattern in patterns:
        found.extend(
            candidate
            for candidate in path.glob(pattern)
            if candidate.is_file() and candidate.stat().st_size > 0
        )
    return sorted(set(found))


def _directory_check(
    name: str,
    path: Path,
    required_files: Iterable[str],
    weight_patterns: Iterable[str],
) -> AssetCheck:
    missing = [
        filename
        for filename in required_files
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0
    ]
    weights = _files(path, weight_patterns)
    if not path.is_dir():
        return AssetCheck(name, "missing", str(path), "directory does not exist")
    if missing:
        return AssetCheck(name, "invalid", str(path), "missing: " + ", ".join(missing))
    if not weights:
        return AssetCheck(name, "invalid", str(path), "no model weight file found")
    return AssetCheck(name, "ok", str(path), f"{len(weights)} weight file(s)")


def verify_assets(project_root: Path) -> Dict[str, object]:
    root = Path(project_root).resolve()
    checks = [
        _directory_check(
            "ga_vln_checkpoint",
            root / "checkpoints" / "gavln_official",
            ("config.json", "model.safetensors.index.json"),
            ("*.safetensors",),
        ),
        _directory_check(
            "siglip2",
            root / "model" / "siglip-so400m-patch14-384",
            ("config.json",),
            ("model.safetensors",),
        ),
        _directory_check(
            "vggt",
            root / "model" / "VGGT-1B",
            ("config.json",),
            ("model.safetensors",),
        ),
    ]

    r2r_root = root / "vln_data" / "datasets" / "r2r"
    r2r_splits = ("train", "val_seen", "val_unseen")
    missing_r2r = [
        split
        for split in r2r_splits
        if not (r2r_root / split / f"{split}.json.gz").is_file()
        or (r2r_root / split / f"{split}.json.gz").stat().st_size == 0
    ]
    checks.append(
        AssetCheck(
            "r2r_vlnce",
            "ok" if not missing_r2r else ("missing" if not r2r_root.exists() else "invalid"),
            str(r2r_root),
            "required splits present"
            if not missing_r2r
            else "missing splits: " + ", ".join(missing_r2r),
        )
    )

    scene_root = root / "vln_data" / "scene_datasets" / "mp3d"
    scene_files = (
        sorted(
            path
            for path in scene_root.glob("*/*.glb")
            if path.is_file() and path.stat().st_size > 0
        )
        if scene_root.is_dir()
        else []
    )
    checks.append(
        AssetCheck(
            "matterport3d_habitat",
            "ok" if len(scene_files) == 90 else ("missing" if not scene_root.exists() else "invalid"),
            str(scene_root),
            f"{len(scene_files)}/90 .glb scenes",
        )
    )

    rxr_candidates = (
        root / "vln_data" / "datasets" / "RxR_VLNCE_v0",
        root / "vln_data" / "datasets" / "rxr",
    )
    rxr_root = next((path for path in rxr_candidates if path.is_dir()), rxr_candidates[0])
    guide_files = (
        sorted(
            path
            for path in rxr_root.glob("*/*_guide.json.gz")
            if path.is_file() and path.stat().st_size > 0
        )
        if rxr_root.is_dir()
        else []
    )
    checks.append(
        AssetCheck(
            "rxr_vlnce",
            "ok" if guide_files else ("missing" if not rxr_root.exists() else "invalid"),
            str(rxr_root),
            f"{len(guide_files)} guide split file(s)",
        )
    )

    payload_checks = [asdict(check) for check in checks]
    required = {"ga_vln_checkpoint", "siglip2", "vggt", "r2r_vlnce", "matterport3d_habitat"}
    ready = all(check.status == "ok" for check in checks if check.name in required)
    return {
        "project_root": str(root),
        "baseline_ready": ready,
        "checks": payload_checks,
    }


def format_asset_report(report: Mapping[str, object]) -> str:
    lines = [f"Project root: {report['project_root']}"]
    for item in report["checks"]:  # type: ignore[index]
        check = item  # type: ignore[assignment]
        lines.append(
            "[{status:7}] {name:22} {detail} ({path})".format(**check)
        )
    lines.append(f"Baseline ready: {report['baseline_ready']}")
    return "\n".join(lines)
