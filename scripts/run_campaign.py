#!/usr/bin/env python3
"""Run the four GB300 experiments as one pilot or final campaign."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    "memory_paths": 540,
    "umma_throughput": 720,
    "umma_device_scaling": 120,
    "gemm_comparison": 20,
}
SCALING_CONFIGURATIONS = {
    ("umma_1sm", "isolated"), ("umma_2sm", "isolated"),
    ("umma_1sm", "device_scale"), ("umma_2sm", "device_scale"),
}


def run(command):
    print(f"campaign: {' '.join(map(str, command))}", file=sys.stderr, flush=True)
    return subprocess.run(command, cwd=ROOT, check=True, text=True)


def gpu_info(index):
    result = subprocess.run(
        ["nvidia-smi", "-i", index, "--query-gpu=uuid,name,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, check=True,
    )
    values = next(csv.reader(result.stdout.splitlines()))
    if len(values) != 3:
        raise RuntimeError("nvidia-smi did not identify one GPU")
    return dict(zip(("uuid", "name", "driver_version"), (value.strip() for value in values)))


def check_csv(path, experiment):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = EXPECTED_ROWS[experiment]
    if len(rows) != expected:
        raise RuntimeError(f"{path.name}: expected {expected} rows, received {len(rows)}")
    for row in rows:
        if row.get("correctness") not in ("OK", "PASS"):
            raise RuntimeError(f"{path.name}: numerical validation did not pass")
    if experiment == "umma_device_scaling":
        configurations = {(row["method"], row["scale"]) for row in rows}
        if configurations != SCALING_CONFIGURATIONS:
            raise RuntimeError("device-scale UMMA did not execute all four configurations")
        for row in rows:
            if row["scale"] == "device_scale":
                if int(row["observed_unique_sm_count"]) != int(row["planned_active_sm_count"]):
                    raise RuntimeError("device-scale UMMA did not observe its planned SM coverage")
                if row["residency_evidence"] != "all_blocks_simultaneously_resident":
                    raise RuntimeError("device-scale UMMA did not establish simultaneous residency")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Run one complete GB300 campaign.")
    parser.add_argument("--kind", required=True, choices=("pilot", "final"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--with-ncu", action="store_true")
    args = parser.parse_args()

    gpu_index = os.environ.get("BLACKWELL_GPU_INDEX", "")
    if not gpu_index.isdigit():
        raise SystemExit("set BLACKWELL_GPU_INDEX")
    now = dt.datetime.now(dt.timezone.utc)
    campaign_id = args.campaign_id or now.strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    root = root.resolve()
    if not root.is_relative_to(ROOT):
        raise RuntimeError("campaigns must remain inside the mounted repository")
    directory = root / campaign_id
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=False)

    commands = (
        ("memory_paths", [sys.executable, "memory_paths/benchmark.py", "--run-kind", "benchmark"]),
        ("umma_throughput", [sys.executable, "umma_throughput/benchmark.py", "--run-kind", "benchmark"]),
        ("umma_device_scaling", [sys.executable, "umma_throughput/benchmark.py", "--device-scaling",
                                 "--run-kind", "benchmark", "--campaign-kind", args.kind]),
    )
    counts = {}
    for experiment, command in commands:
        path = raw / f"{experiment}.csv"
        run([*command, "--output", str(path)])
        counts[experiment] = check_csv(path, experiment)

    gemm_path = raw / "gemm_comparison.csv"
    relative = os.path.relpath(gemm_path, ROOT / "gemm_comparison")
    run([str(ROOT / "scripts/run_gpu.sh"), "make", "-C", "gemm_comparison", "run",
         f"ARCH={os.environ.get('CUDA_ARCH', 'sm_103a')}", f"OUTPUT={relative}",
         "WARMUP=2", "ITERATIONS=10"])
    counts["gemm_comparison"] = check_csv(gemm_path, "gemm_comparison")

    profile = {"state": "NOT_REQUESTED", "captured_count": 0}
    if args.with_ncu:
        import ncu_capture
        profile = ncu_capture.capture(directory)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout.strip()
    metadata = {
        "campaign_id": campaign_id,
        "kind": args.kind,
        "created_utc": now.isoformat(),
        "git_commit": commit or "unknown",
        "gpu": gpu_info(gpu_index),
        "container_image": os.environ.get("IMAGE_TAG", "gb300-microbench:latest"),
        "row_counts": counts,
        "ncu": {key: profile[key] for key in ("state", "captured_count")},
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"campaign: COMPLETE {directory}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"campaign: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
