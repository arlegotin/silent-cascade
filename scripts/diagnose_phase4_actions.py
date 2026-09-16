"""Authenticate original Phase3 artifacts and diagnose four action contexts offline."""

import argparse
import json
from pathlib import Path

from silent_cascade.errors import SilentCascadeError
from silent_cascade.eval.action_diagnostics import _authenticate_files, _diagnose_authenticated
from silent_cascade.train.pilot_config import resolve_pilot_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--profile", choices=("phase4_pilot", "phase4_smoke"), default="phase4_pilot"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        restored, manifest, examples = _authenticate_files(
            source_weights_path=args.weights.absolute(),
            expected_source_weights_sha256=args.expected_weights_sha256,
            source_manifest_path=args.manifest.absolute(),
            expected_source_manifest_sha256=args.expected_manifest_sha256,
            device=args.device,
        )
        report = _diagnose_authenticated(
            restored=restored,
            manifest=manifest,
            examples=examples,
            config=resolve_pilot_config(args.profile).config,
            manifest_sha256=args.expected_manifest_sha256,
            output_dir=args.output.absolute(),
        )
        print(
            json.dumps(
                {
                    "diagnostic_only": report.diagnostic_only,
                    "historical_reproduction": report.historical_reproduction,
                    "episode_count": report.autonomous.episode_count,
                    "timed_success_count": report.autonomous.timed_success_count,
                    "elapsed_seconds": report.elapsed_seconds,
                }
            )
        )
        return 0
    except (SilentCascadeError, ValueError, OSError) as error:
        print(json.dumps({"error": "action_diagnostic_error", "message": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
