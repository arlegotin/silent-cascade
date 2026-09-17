"""Run the selected-weight final checks without repeating pilot training."""

import argparse
from pathlib import Path

from silent_cascade.train.pilot_checks import recover_pilot_checks, run_pilot_checks
from silent_cascade.train.pilot_workflow import resolve_pilot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--retained-run-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.recover_from is not None:
        if args.config is not None or args.output is not None or args.destination is None:
            parser.error("recovery requires --destination and forbids --config/--output")
        artifact = recover_pilot_checks(
            artifact_path=args.recover_from,
            raw_run_dir=args.run_dir,
            retained_run_dir=args.retained_run_dir,
            destination=args.destination,
        )
        print(f"Recovery execution DONE; final gate={artifact.outcome}")
        return int(artifact.outcome == "failed")
    if (
        args.config is None
        or args.output is None
        or args.retained_run_dir is not None
        or args.destination is not None
    ):
        parser.error("normal mode requires --config/--output and forbids recovery-only flags")
    config = resolve_pilot_path(args.config)
    artifact = run_pilot_checks(run_dir=args.run_dir, config=config, output_path=args.output)
    print(f"Execution DONE; final gate={artifact.outcome}")
    return int(config.config.pilot.is_production and artifact.outcome != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
