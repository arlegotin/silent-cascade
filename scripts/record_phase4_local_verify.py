"""Record the existing full local make verify gate, without scientific execution."""

import argparse
from pathlib import Path

from silent_cascade.train.pilot_checks import record_local_verification
from silent_cascade.train.provenance import git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    output = root / "artifacts/phase4-local-verification" / revision
    receipt = record_local_verification(repo_root=root, output_dir=output)
    print(f"Local verification receipt: {output / 'receipt.json'}")
    return int(receipt.returncode != 0 or not receipt.source_unchanged)


if __name__ == "__main__":
    raise SystemExit(main())
