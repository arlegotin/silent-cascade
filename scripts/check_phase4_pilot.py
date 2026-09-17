"""Run the selected-weight final checks without repeating pilot training."""

import argparse
from pathlib import Path

from silent_cascade.train.pilot_checks import run_pilot_checks
from silent_cascade.train.pilot_workflow import resolve_pilot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_pilot_path(args.config)
    artifact = run_pilot_checks(run_dir=args.run_dir, config=config, output_path=args.output)
    print(f"Execution DONE; final gate={artifact.outcome}")
    return int(config.config.pilot.is_production and artifact.outcome != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
