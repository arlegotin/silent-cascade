"""Local one-seed workflow; immutable data must be introduced before fitting."""

import argparse
from pathlib import Path

from silent_cascade.errors import SilentCascadeError
from silent_cascade.train.pilot_workflow import resolve_pilot_path, run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    try:
        result = run_pilot(
            config_path=args.config,
            manifest_dir=args.manifest_dir,
            run_dir=args.run_dir,
            device=args.device,
        )
    except (ValueError, OSError, RuntimeError, SilentCascadeError) as error:
        parser.exit(
            1,
            f"Pilot workflow refused: {error}\n"
            "For incompatible directories, choose a new-run-path and preserve prior evidence.\n",
        )
    production = resolve_pilot_path(args.config).config.pilot.is_production
    print(
        f"Pilot execution completed: {result.progress.global_step} updates; "
        f"production learning gate={'passed' if result.gate_eligible else 'unmet'}"
    )
    return 1 if production and not result.gate_eligible else 0


if __name__ == "__main__":
    raise SystemExit(main())
