from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_jobs(project_root: Path) -> list[tuple[Path, list[str]]]:
    """Build the list of runner invocations needed to regenerate results."""
    runners = [
        project_root / "src" / "runner.py",
        project_root / "src" / "different_vision_cones" / "runner.py",
    ]

    jobs: list[tuple[Path, list[str]]] = []
    for runner in runners:
        for mode in ("selfish", "auxiliary"):
            args = ["--mode", mode]
            if runner.parent.name == "different_vision_cones":
                args.extend(["--theta_deg2", "90"])
                args.extend(["--cone_assign", "rows"])
            jobs.append((runner, args))
    return jobs


def run_jobs(jobs: list[tuple[Path, list[str]]], dry_run: bool) -> int:
    """Run each runner command in sequence and stop on first failure."""
    total = len(jobs)

    for idx, (runner, args) in enumerate(jobs, start=1):
        cmd = [sys.executable, str(runner), *args]
        cwd = runner.parent

        print(f"[{idx}/{total}] Running in {cwd}")
        print("  ", " ".join(cmd))

        if dry_run:
            continue

        completed = subprocess.run(cmd, cwd=cwd)
        if completed.returncode != 0:
            print(f"Command failed with exit code {completed.returncode}")
            return completed.returncode

    print("All runner jobs completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate project results by running all runner.py files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    jobs = build_jobs(project_root)
    return run_jobs(jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
