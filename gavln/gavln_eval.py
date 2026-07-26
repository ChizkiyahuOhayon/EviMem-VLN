from __future__ import annotations

import sys
from typing import Optional, Sequence

from evimem.assets import format_asset_report, verify_assets
from gavln.cli import PROJECT_ROOT, parse_args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = verify_assets(PROJECT_ROOT)
    if not report["baseline_ready"]:
        print(format_asset_report(report), file=sys.stderr)
        print(
            "Required runtime assets are missing. See docs/DATA.md or run "
            "`evimem verify-assets`.",
            file=sys.stderr,
        )
        return 2

    from gavln.eval_runtime import run

    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
