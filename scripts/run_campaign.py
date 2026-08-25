#!/usr/bin/env python3
"""Run the four experiments inside one GPU-selected container."""

import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(f"campaign: {' '.join(map(str, command))}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return list(csv.DictReader(io.StringIO(completed.stdout)))


def write_rows(path, rows, expected):
    # Persist only complete datasets whose numerical checks passed.
    if len(rows) != expected or any(row.get("correctness") not in ("OK", "PASS") for row in rows):
        raise RuntimeError(f"{path.name}: expected {expected} valid samples")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def gpu_info():
    gpu = os.environ.get("BLACKWELL_GPU_UUID", "0")
    completed = subprocess.run(
        ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,driver_version",
         "--format=csv,noheader"], text=True, capture_output=True, check=True)
    return dict(zip(("uuid", "name", "driver_version"),
                    (value.strip() for value in next(csv.reader(completed.stdout.splitlines())))))


def memory_rows(kind):
    # Pilots shorten the workload without changing the experimental sweep.
    working_set, passes, warmup, repetitions = (64, 2, 200, 5) if kind == "pilot" else (512, 32, 2000, 30)
    run_kind = "smoke" if kind == "pilot" else "benchmark"
    rows = []
    for stages in (2, 4, 8):
        for in_flight in (16, 32, 64):
            for method in ("ldgsts", "tma"):
                rows.extend(run([
                    f"build/memory_paths/{method}", "--stages", str(stages),
                    "--bytes-in-flight-kib", str(in_flight), "--run-kind", run_kind,
                    "--working-set-mib", str(working_set), "--passes", str(passes),
                    "--warmup-ms", str(warmup), "--repetitions", str(repetitions)]))
    return rows, 18 * repetitions


def umma_rows(kind):
    iterations, warmup, repetitions = (20, 5, 3) if kind == "pilot" else (1000, 10, 30)
    run_kind = "smoke" if kind == "pilot" else "benchmark"
    common = ["--run-kind", run_kind, "--iterations", str(iterations),
              "--warmup-iterations", str(warmup), "--repetitions", str(repetitions)]
    rows = []
    for n in (64, 128, 256):
        for depth in (4, 16, 64, 256):
            for method in ("umma_1sm", "umma_2sm"):
                rows.extend(run([f"build/umma_throughput/{method}", *common,
                                 "--n", str(n), "--depth", str(depth)]))
    scaling = run(["build/umma_throughput/umma_device_scaling", *common,
                   "--campaign-kind", kind])
    return rows, 24 * repetitions, scaling, 4 * repetitions


def main():
    parser = argparse.ArgumentParser(description="Run one GB300 pilot or final campaign.")
    parser.add_argument("--kind", required=True, choices=("pilot", "final"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--with-ncu", action="store_true")
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    campaign_id = args.campaign_id or now.strftime("%Y%m%dT%H%M%SZ")
    directory = (args.output_root if args.output_root.is_absolute() else ROOT / args.output_root) / campaign_id
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=False)

    memory, memory_count = memory_rows(args.kind)
    umma, umma_count, scaling, scaling_count = umma_rows(args.kind)
    gemm_warmup, gemm_iterations = (1, 1) if args.kind == "pilot" else (2, 10)
    gemm = run([sys.executable, "gemm_comparison/gemm_comparison.py",
                "--warmup-iterations", str(gemm_warmup), "--iterations", str(gemm_iterations)])
    counts = {
        "memory_paths": write_rows(raw / "memory_paths.csv", memory, memory_count),
        "umma_throughput": write_rows(raw / "umma_throughput.csv", umma, umma_count),
        "umma_device_scaling": write_rows(raw / "umma_device_scaling.csv", scaling, scaling_count),
        "gemm_comparison": write_rows(raw / "gemm_comparison.csv", gemm, 20),
    }

    profile = {"state": "NOT_REQUESTED", "captured_count": 0}
    if args.with_ncu:
        # Profile after timing so NCU replay cannot affect measured throughput.
        import ncu_capture
        profile = ncu_capture.capture(directory)

    metadata = {"campaign_id": campaign_id, "kind": args.kind, "created_utc": now.isoformat(),
                "gpu": gpu_info(), "row_counts": counts,
                "ncu": {key: profile[key] for key in ("state", "captured_count")}}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"campaign: COMPLETE {directory}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"campaign: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
