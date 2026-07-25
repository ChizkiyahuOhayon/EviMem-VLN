from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .assets import format_asset_report, verify_assets


PINNED_GA_VLN_COMMIT = "cc6086b7081a346695abecf6821f827e2db44a43"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def doctor(project_root: Path = PROJECT_ROOT) -> Dict[str, object]:
    root = Path(project_root).resolve()
    assets = verify_assets(root)
    source_ready = (root / "gavln" / "model" / "model_gavln.py").is_file()
    return {
        "python": sys.version.split()[0],
        "project_root": str(root),
        "upstream_ga_vln_commit": PINNED_GA_VLN_COMMIT,
        "internal_source_ready": source_ready,
        "assets": assets,
        "ready": source_ready and bool(assets["baseline_ready"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evimem")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "verify-assets"):
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        child.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-assets":
        report = verify_assets(args.project_root)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_asset_report(report))
        return 0 if report["baseline_ready"] else 2

    report = doctor(args.project_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Python: {report['python']}")
        print(f"Internal GA-VLN source: {report['internal_source_ready']}")
        print(f"Upstream commit: {report['upstream_ga_vln_commit']}")
        print(format_asset_report(report["assets"]))  # type: ignore[arg-type]
        print(f"Ready: {report['ready']}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
