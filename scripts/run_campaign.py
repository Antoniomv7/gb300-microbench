#!/usr/bin/env python3
"""Run Experiments I–IV with the final parameters inside one GPU-selected container.

A campaign runs all four experiments and the eight Nsight Compute captures. --experiments runs a
subset for the per-experiment Makefile targets, with only that subset's own captures.
"""

import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import gpu_telemetry
import ncu_capture
import provenance

ROOT = Path(__file__).resolve().parents[1]
# Final measurement parameters of every campaign and single-experiment run.
MEMORY = {"working_set_mib": 512, "passes": 32, "warmup_ms": 2000, "repetitions": 30}
UMMA = {"iterations": 1000, "warmup_iterations": 10, "repetitions": 30}
GEMM = {"warmup_iterations": 2, "iterations": 10}
TELEMETRY_INTERVAL_MS = 50
MINIMUM_SAMPLES_PER_CONFIGURATION = 3
SOURCES = ("benchmark_common.cuh", "memory_paths/memory_common.cuh", "memory_paths/ldgsts.cu",
           "memory_paths/tma.cu", "umma_throughput/umma_common.cuh", "umma_throughput/umma_1sm.cu",
           "umma_throughput/umma_2sm.cu", "umma_throughput/umma_device_scaling.cu",
           "gemm_comparison/gemm_comparison.py", "gemm_comparison/cublaslt_bridge.cu",
           "scripts/run_campaign.py", "scripts/gpu_telemetry.py", "scripts/ncu_capture.py",
           "scripts/provenance.py", "scripts/run_gpu.sh")


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


def memory_paths(raw):
    rows = []
    for stages in (2, 4, 8):
        for in_flight in (16, 32, 64):
            for method in ("ldgsts", "tma"):
                rows.extend(run([
                    f"build/memory_paths/{method}", "--stages", str(stages),
                    "--bytes-in-flight-kib", str(in_flight), "--run-kind", "benchmark",
                    "--working-set-mib", str(MEMORY["working_set_mib"]),
                    "--passes", str(MEMORY["passes"]), "--warmup-ms", str(MEMORY["warmup_ms"]),
                    "--repetitions", str(MEMORY["repetitions"])]))
    return {"rows": write_rows(raw / "memory_paths.csv", rows, 18 * MEMORY["repetitions"])}


UMMA_ARGUMENTS = ["--run-kind", "benchmark", "--iterations", str(UMMA["iterations"]),
                  "--warmup-iterations", str(UMMA["warmup_iterations"]),
                  "--repetitions", str(UMMA["repetitions"])]


def umma_throughput(raw):
    rows = []
    for n in (64, 128, 256):
        for depth in (4, 16, 64, 256):
            for method in ("umma_1sm", "umma_2sm"):
                rows.extend(run([f"build/umma_throughput/{method}", *UMMA_ARGUMENTS,
                                 "--n", str(n), "--depth", str(depth)]))
    return {"rows": write_rows(raw / "umma_throughput.csv", rows, 24 * UMMA["repetitions"])}


def telemetry_overlap(rows, samples):
    # A sample belongs to a configuration when it falls inside one of its timed launches.
    windows = defaultdict(list)
    for row in rows:
        windows[f"{row['method']}/{row['scale']}"].append(
            (float(row["host_start_unix_s"]), float(row["host_end_unix_s"])))
    moments = [sample[0] for sample in samples]
    starts = [start for spans in windows.values() for start, _ in spans]
    ends = [end for spans in windows.values() for _, end in spans]
    if not moments or min(moments) > min(starts) or max(moments) < max(ends):
        raise RuntimeError("clock sampling did not cover the whole timed campaign")
    return {key: sum(1 for moment in moments
                     if any(start <= moment <= end for start, end in spans))
            for key, spans in sorted(windows.items())}


def umma_device_scaling(raw):
    """Time the four scaling configurations while sampling the SM clock alongside."""
    gpu = os.environ.get("BLACKWELL_GPU_UUID", "0")
    with gpu_telemetry.ClockSampler(gpu, TELEMETRY_INTERVAL_MS) as sampler:
        rows = run(["build/umma_throughput/umma_device_scaling", *UMMA_ARGUMENTS])
    summary = sampler.verify()
    summary["written_count"] = sampler.write(raw / "umma_device_scaling_telemetry.csv")
    summary["samples_per_configuration"] = telemetry_overlap(rows, sampler.samples)
    if min(summary["samples_per_configuration"].values()) < MINIMUM_SAMPLES_PER_CONFIGURATION:
        raise RuntimeError(f"too few clock samples: {summary['samples_per_configuration']}")
    count = write_rows(raw / "umma_device_scaling.csv", rows, 4 * UMMA["repetitions"])
    return {"rows": count, "telemetry": {**summary, "state": "COMPLETE"}}


def gemm_comparison(raw):
    rows = run([sys.executable, "gemm_comparison/gemm_comparison.py",
                "--warmup-iterations", str(GEMM["warmup_iterations"]),
                "--iterations", str(GEMM["iterations"])])
    return {"rows": write_rows(raw / "gemm_comparison.csv", rows, 20)}


EXPERIMENTS = {"memory_paths": memory_paths, "umma_throughput": umma_throughput,
               "umma_device_scaling": umma_device_scaling, "gemm_comparison": gemm_comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS),
                        help="comma-separated subset of " + ",".join(EXPERIMENTS))
    args = parser.parse_args()
    selected = {name.strip() for name in args.experiments.split(",") if name.strip()}
    if not selected or selected - set(EXPERIMENTS):
        parser.error("--experiments must name a subset of " + ",".join(EXPERIMENTS))
    experiments = tuple(name for name in EXPERIMENTS if name in selected)

    now = dt.datetime.now(dt.timezone.utc)
    root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    directory = root / args.campaign_id
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / "raw"
    raw.mkdir()
    # Identify the GPU and the exact sources before any measurement.
    environment = {"gpu": provenance.gpu_identity(), "software": provenance.software_versions(),
                   "repository": provenance.repository_state(SOURCES),
                   "pinned": provenance.pinned_versions()}

    counts, telemetry = {}, {"state": "NOT_RUN"}
    for name in experiments:
        result = EXPERIMENTS[name](raw)
        counts[name] = result["rows"]
        telemetry = result.get("telemetry", telemetry)

    # Profile after timing so NCU replay cannot affect measured throughput.
    cases = ncu_capture.planned_cases(experiments)
    profile = {"state": "NONE_PLANNED", "captured_count": 0, "cases": []}
    if cases:
        profile = ncu_capture.capture(directory, cases)
        environment["ncu"] = provenance.ncu_version()

    metadata = {"campaign_id": args.campaign_id, "kind": "final", "state": "COMPLETE",
                "created_utc": now.isoformat(),
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "gpu": environment["gpu"], "environment": environment,
                "experiments": list(experiments), "row_counts": counts,
                "parameters": {"memory_paths": MEMORY, "umma": UMMA, "gemm_comparison": GEMM,
                               "telemetry_interval_ms": TELEMETRY_INTERVAL_MS},
                "telemetry": telemetry,
                "ncu": {"state": profile["state"], "captured_count": profile["captured_count"],
                        "cases": [case["case"] for case in profile["cases"]]}}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"campaign: COMPLETE {directory}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"campaign: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
