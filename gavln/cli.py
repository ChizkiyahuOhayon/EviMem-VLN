from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gavln-eval",
        description="Evaluate the integrated GA-VLN or EviMem memory backend.",
    )
    parser.add_argument("--config", type=Path, help="Optional YAML file with argument defaults.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "gavln_official"),
    )
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default=str(PROJECT_ROOT / "config" / "vln_r2r.yaml"),
    )
    parser.add_argument("--eval_split", type=str, default="val_unseen")
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(PROJECT_ROOT / "results" / "val_unseen" / "gavln"),
    )
    parser.add_argument(
        "--vision_tower_path",
        type=str,
        default=str(PROJECT_ROOT / "model" / "siglip-so400m-patch14-384"),
    )
    parser.add_argument(
        "--vggt_path",
        type=str,
        default=str(PROJECT_ROOT / "model" / "VGGT-1B"),
    )
    parser.add_argument("--bev_num_steps", type=int, default=4)
    parser.add_argument("--bev_max_steps", type=int, default=32)
    parser.add_argument("--bev_grid_size", type=float, default=0.25)
    parser.add_argument("--bev_range", type=float, default=10.0)
    parser.add_argument("--bev_pos_temp", type=float, default=10000)
    parser.add_argument("--num_front_view", type=int, default=0)
    parser.add_argument("--dia_round", type=int, default=2)
    parser.add_argument("--patch_grid_size", type=int, default=27)
    parser.add_argument("--patch_size_pixel", type=int, default=14)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--memory_backend",
        choices=("gavln", "evimem"),
        default="gavln",
        help="Use the original windowed BEV or persistent EviMem sparse world memory.",
    )
    parser.add_argument(
        "--memory_horizon",
        choices=("8", "32", "64", "route"),
        default="32",
        help="Executed-action retention horizon for EviMem.",
    )
    parser.add_argument("--memory_resident_slots", type=int, default=2048)
    parser.add_argument(
        "--memory_token_budget",
        type=int,
        default=None,
        help="Maximum memory tokens per policy refresh; unset preserves all non-empty cells.",
    )
    parser.add_argument("--memory_seed", type=int, default=0)
    return parser


def _yaml_defaults(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required when --config is used") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must contain a YAML mapping")
    unknown = set(payload) - {action.dest for action in build_parser()._actions}
    if unknown:
        raise ValueError("unknown config keys: " + ", ".join(sorted(unknown)))
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    known, _ = config_parser.parse_known_args(argv)
    parser = build_parser()
    if known.config is not None:
        parser.set_defaults(**_yaml_defaults(known.config))
    args = parser.parse_args(argv)
    if args.memory_resident_slots <= 0:
        parser.error("--memory_resident_slots must be positive")
    if args.memory_token_budget is not None and args.memory_token_budget <= 0:
        parser.error("--memory_token_budget must be positive")
    args.memory_horizon = (
        args.memory_horizon if args.memory_horizon == "route" else int(args.memory_horizon)
    )
    return args
