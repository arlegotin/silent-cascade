"""Independently verify retained Phase4 evidence without neural execution."""

import argparse
import json
from pathlib import Path

from silent_cascade.train.pilot_evidence import verify_phase4_gate_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-run-dir", type=Path)
    args = parser.parse_args()
    result = verify_phase4_gate_artifact(
        args.artifact, repo_root=args.repo_root, raw_run_dir=args.raw_run_dir
    )
    print(json.dumps(result, sort_keys=True))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
