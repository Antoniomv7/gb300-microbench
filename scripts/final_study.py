#!/usr/bin/env python3
"""Run the final study: Experiments I–IV three times, Experiment V once, and the hot-cache profile.

Every step is one of the Makefile's own commands, so a failed step can be rerun alone for
diagnosis. The study stops at the first failure, never reuses a directory, and archives its
evidence only after scripts/check_diagnostics.py has passed the whole study.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import provenance

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ("campaign-1", "campaign-2", "campaign-3")
ANALYSIS = "analysis"
PRECISION = "precision"
PROFILE = "gemm-profile-hot"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def steps(study):
    """The study's commands in order; study is relative to the repository root."""
    make = ["make", "--no-print-directory"]
    check = [sys.executable, "scripts/check_diagnostics.py"]
    reference = study / ANALYSIS / "gemm_comparison.csv"
    return [
        *((name, [*make, "campaign", f"RUNS={study}", f"CAMPAIGN_ID={name}"])
          for name in CAMPAIGNS),
        ("check-campaigns", [*check, "--final-campaigns",
                             *(str(study / name) for name in CAMPAIGNS)]),
        (ANALYSIS, [*make, "analyze", f"RUNS={study}", f"FINAL_CAMPAIGNS={' '.join(CAMPAIGNS)}",
                    f"ANALYSIS_OUT={study / ANALYSIS}"]),
        (PRECISION, [*make, "precision", f"RUNS={study}", f"PRECISION_ID={PRECISION}"]),
        # The new analysis, never results/, is the profile's CUDA-event reference.
        (PROFILE, [*make, "gemm-profile", f"RUNS={study}", f"PROFILE_ID={PROFILE}",
                   "PROFILE_CACHE=hot", f"GEMM_SUMMARY={reference}"]),
        ("check-study", [*check, "--study", str(study)]),
    ]


def run_logged(command, log):
    """Run one step, mirroring its output to the terminal and to its log."""
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return process.wait()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="parent of the new study directory, inside the repository")
    args = parser.parse_args()
    if not os.environ.get("BLACKWELL_GPU_INDEX", "").isdigit():
        raise SystemExit("final-study: set BLACKWELL_GPU_INDEX to the B300 to use")
    repository = provenance.repository_state()
    if provenance.tracked_changes(repository):
        raise SystemExit("final-study: commit or discard the modified tracked files first; every "
                         f"run records its source commit: {provenance.tracked_changes(repository)}")
    created = dt.datetime.now(dt.timezone.utc)
    study_id = created.strftime("study-%Y%m%dT%H%M%SZ")
    directory = (args.runs if args.runs.is_absolute() else ROOT / args.runs).resolve() / study_id
    # The containers mount only the repository, so every path they receive is relative to it.
    if not directory.is_relative_to(ROOT):
        raise SystemExit(f"final-study: {args.runs} must lie inside {ROOT}")
    study = directory.relative_to(ROOT)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "logs").mkdir()
    record = {"study_id": study_id, "state": "RUNNING", "created_utc": created.isoformat(),
              "source_commit": repository["commit"], "steps": []}

    def save(state, **fields):
        record.update(state=state, completed_utc=now(), **fields)
        (directory / "study.json").write_text(json.dumps(record, indent=2) + "\n",
                                              encoding="utf-8")

    plan = steps(study)
    for number, (name, command) in enumerate(plan, 1):
        log = directory / "logs" / f"{number:02d}-{name}.log"
        print(f"final-study: [{number}/{len(plan)}] {name}: {' '.join(command)}", flush=True)
        entry = {"name": name, "command": command, "log": str(log.relative_to(directory)),
                 "started_utc": now()}
        record["steps"].append(entry)
        try:
            status = run_logged(command, log)
        except KeyboardInterrupt:
            save("INTERRUPTED", failed_step=name)
            raise SystemExit(f"final-study: interrupted during {name}; {study} is kept, not reused")
        entry.update(finished_utc=now(), returncode=status)
        if status:
            save("FAILED", failed_step=name)
            raise SystemExit(f"final-study: {name} failed with status {status}; see {log}. "
                             f"Nothing was archived and {study} is not reused.")

    if provenance.command_output(["git", "rev-parse", "HEAD"]) != repository["commit"]:
        save("FAILED", failed_step="source commit changed during the study")
        raise SystemExit("final-study: HEAD changed during the study; nothing was archived")
    metadata = directory / CAMPAIGNS[0] / "metadata.json"
    gpu = json.loads(metadata.read_text(encoding="utf-8"))["gpu"]
    archive = directory.with_name(f"{study_id}.tar.gz")
    save("COMPLETE", gpu=gpu, archive=str(archive.relative_to(ROOT)))
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(directory, arcname=study_id)
    print(f"final-study: COMPLETE {study}\n"
          f"  archive:       {archive.relative_to(ROOT)} (sha256 {sha256(archive)})\n"
          f"  GPU:           {gpu['uuid']} ({gpu['name']})\n"
          f"  source commit: {repository['commit']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"final-study: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
