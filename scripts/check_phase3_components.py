"""Freeze a reviewed component recipe or collect real selected-weight evidence offline."""

import argparse
import json
from pathlib import Path

from silent_cascade.config import resolve_config
from silent_cascade.errors import SilentCascadeError
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.evidence import collect_component_gate
from silent_cascade.train.provenance import freeze_component_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("freeze-validation", "collect"):
        command = commands.add_parser(mode)
        command.add_argument("--config", action="append", required=True, type=Path)
        command.add_argument("--expected-source-commit", required=True)
        command.add_argument("--expected-plan-base-revision", required=True)
        command.add_argument("--output", required=True, type=Path)
        if mode == "collect":
            command.add_argument("--manifest", required=True, type=Path)
            command.add_argument("--weights", required=True, type=Path)
            command.add_argument("--expected-checkpoint-sha256", required=True)
            command.add_argument("--training-run", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = resolve_config(Phase3Config, tuple(args.config))
        common = dict(
            source_commit=args.expected_source_commit,
            plan_revision=args.expected_plan_base_revision,
            output_path=args.output.absolute(),
        )
        if args.mode == "freeze-validation":
            result = freeze_component_manifest(config, **common)
        else:
            result = collect_component_gate(
                config,
                manifest_path=args.manifest.absolute(),
                weights_path=args.weights.absolute(),
                training_run=args.training_run.absolute(),
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
                **common,
            )
        print(result.model_dump_json())
        return 0
    except (SilentCascadeError, ValueError, OSError) as error:
        print(
            json.dumps(
                {"error": getattr(error, "code", "component_gate_error"), "message": str(error)}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
