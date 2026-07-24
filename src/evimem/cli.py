from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from .assets import format_asset_report, verify_assets


PINNED_GA_VLN_COMMIT = "cc6086b7081a346695abecf6821f827e2db44a43"


def _git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def doctor(ga_vln_root: Path) -> Dict[str, object]:
    root = Path(ga_vln_root).resolve()
    commit = _git_commit(root)
    assets = verify_assets(root)
    return {
        "python": sys.version.split()[0],
        "ga_vln_root": str(root),
        "ga_vln_commit": commit,
        "expected_ga_vln_commit": PINNED_GA_VLN_COMMIT,
        "commit_ok": commit == PINNED_GA_VLN_COMMIT,
        "assets": assets,
        "ready": commit == PINNED_GA_VLN_COMMIT and bool(assets["baseline_ready"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evimem")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "verify-assets"):
        child = subparsers.add_parser(command)
        child.add_argument("--ga-vln-root", type=Path, required=True)
        child.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-assets":
        report = verify_assets(args.ga_vln_root)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_asset_report(report))
        return 0 if report["baseline_ready"] else 2

    report = doctor(args.ga_vln_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Python: {report['python']}")
        print(
            "GA-VLN commit: {} (expected {})".format(
                report["ga_vln_commit"], report["expected_ga_vln_commit"]
            )
        )
        print(format_asset_report(report["assets"]))  # type: ignore[arg-type]
        print(f"Ready: {report['ready']}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
