from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph Barlow Twins for personalized anomaly detection in remote dementia monitoring"
    )
    parser.add_argument("--events_csv", required=True, help="Path to raw sensor event CSV")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--config", default=None, help="Optional YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = run_pipeline(
        events_csv=args.events_csv,
        config=config,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
