"""Collect offline, non-neural runtime engineering evidence on CPU."""

import argparse
from pathlib import Path

from silent_cascade.config import resolve_config
from silent_cascade.errors import SilentCascadeError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.evidence import collect_phase2_gate
from silent_cascade.hashing import canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/primary.yaml"))
    parser.add_argument(
        "--event-flow-config", type=Path, default=Path("configs/model/event_flow.yaml")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-plan-base-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=("production", "debug"), default="production")
    parser.add_argument("--episode-count", type=int, default=10_000)
    args = parser.parse_args()
    try:
        resolved = resolve_config(
            Phase2Config, [args.config, args.data_config, args.event_flow_config]
        )
        report = collect_phase2_gate(
            repo_root=args.repo_root,
            resolved=resolved,
            manifest_path=args.manifest,
            expected_source_commit=args.expected_source_commit,
            expected_plan_base_revision=args.expected_plan_base_revision,
            output_path=args.output,
            profile=args.profile,
            requested_episode_count=args.episode_count,
        )
    except SilentCascadeError as error:
        print(f"Phase 2 gate collection failed: {error.message}")
        return 1
    print(
        canonical_json_bytes(
            {
                "profile": report.profile,
                "passed": report.passed,
                "completed_episode_count": report.completed_episode_count,
                "foundation_model_calls": report.foundation_model_calls,
            }
        ).decode()
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
