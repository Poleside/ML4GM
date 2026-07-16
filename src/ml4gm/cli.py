from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from ml4gm.config import ConfigError, RunConfig
from ml4gm.data.prepare import prepare_dataset
from ml4gm.evaluation.runner import run_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml4gm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--output", default="data/processed/prepared.csv")
    prepare.add_argument("--manifest", default="data/processed/manifest.json")
    for name in ("train", "evaluate", "benchmark"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        return 2
    try:
        config = RunConfig.from_yaml(config_path)
        if args.command == "data":
            prepare_dataset(
                config.data.input,
                Path(args.output),
                Path(args.manifest),
                list(args.source),
                config.data.target,
                config.data.glacier_id,
                config.data.year,
            )
            return 0
        if args.output_dir:
            config = replace(
                config,
                data=replace(config.data, output_dir=Path(args.output_dir)),
            )
        result = run_evaluation(config)
        for fold in result.folds:
            print(
                f"{fold.fold}: R2={fold.r2:.4f} RMSE={fold.rmse:.4f} MAE={fold.mae:.4f}"
            )
        return 0
    except (ConfigError, OSError, ValueError) as exc:
        print(f"ML4GM error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
