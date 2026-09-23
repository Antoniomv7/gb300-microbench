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
from collections import defaultdict
from pathlib import Path

import gpu_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("memory_paths", "umma_throughput", "umma_device_scaling", "gemm_comparison")
TELEMETRY_INTERVAL_MS = 50
MINIMUM_SAMPLES_PER_CONFIGURATION = 3


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


def selected_gpu():
    # run_gpu.sh exposes one physical GPU to the container and names it here.
    return os.environ.get("BLACKWELL_GPU_UUID", "0")


def gpu_info():
    gpu = selected_gpu()
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


def umma_protocol(kind):
    iterations, warmup, repetitions = (20, 5, 3) if kind == "pilot" else (1000, 10, 30)
    run_kind = "smoke" if kind == "pilot" else "benchmark"
    return repetitions, ["--run-kind", run_kind, "--iterations", str(iterations),
                         "--warmup-iterations", str(warmup), "--repetitions", str(repetitions)]


def umma_rows(kind):
    repetitions, common = umma_protocol(kind)
    rows = []
    for n in (64, 128, 256):
        for depth in (4, 16, 64, 256):
            for method in ("umma_1sm", "umma_2sm"):
                rows.extend(run([f"build/umma_throughput/{method}", *common,
                                 "--n", str(n), "--depth", str(depth)]))
    return rows, 24 * repetitions


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


def umma_scaling_rows(kind, telemetry_path):
    """Time the four scaling configurations while sampling the SM clock alongside."""
    repetitions, common = umma_protocol(kind)
    with gpu_telemetry.ClockSampler(selected_gpu(), TELEMETRY_INTERVAL_MS) as sampler:
        rows = run(["build/umma_throughput/umma_device_scaling", *common,
                    "--campaign-kind", kind])
    summary = sampler.verify()
    summary["written_count"] = sampler.write(telemetry_path)
    summary["samples_per_configuration"] = telemetry_overlap(rows, sampler.samples)
    if kind != "pilot" and min(summary["samples_per_configuration"].values()) < \
            MINIMUM_SAMPLES_PER_CONFIGURATION:
        raise RuntimeError(f"too few clock samples: {summary['samples_per_configuration']}")
    return rows, 4 * repetitions, summary


def main():
    parser = argparse.ArgumentParser(description="Run one GB300 pilot or final campaign.")
    parser.add_argument("--kind", required=True, choices=("pilot", "final"))
    parser.add_argument("--campaign-id")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--with-ncu", action="store_true")
    parser.add_argument("--experiments", default=",".join(DATASETS),
                        help="comma-separated subset of " + ",".join(DATASETS))
    args = parser.parse_args()
    experiments = tuple(name.strip() for name in args.experiments.split(",") if name.strip())
    if any(name not in DATASETS for name in experiments) or not experiments:
        parser.error("--experiments must name a subset of " + ",".join(DATASETS))

    now = dt.datetime.now(dt.timezone.utc)
    campaign_id = args.campaign_id or now.strftime("%Y%m%dT%H%M%SZ")
    directory = (args.output_root if args.output_root.is_absolute() else ROOT / args.output_root) / campaign_id
    raw = directory / "raw"
    raw.mkdir(parents=True, exist_ok=False)

    counts, telemetry = {}, {"state": "NOT_REQUESTED"}
    if "memory_paths" in experiments:
        memory, memory_count = memory_rows(args.kind)
        counts["memory_paths"] = write_rows(raw / "memory_paths.csv", memory, memory_count)
    if "umma_throughput" in experiments:
        umma, umma_count = umma_rows(args.kind)
        counts["umma_throughput"] = write_rows(raw / "umma_throughput.csv", umma, umma_count)
    if "umma_device_scaling" in experiments:
        scaling, scaling_count, telemetry = umma_scaling_rows(
            args.kind, raw / "umma_device_scaling_telemetry.csv")
        telemetry["state"] = "COMPLETE"
        counts["umma_device_scaling"] = write_rows(raw / "umma_device_scaling.csv",
                                                   scaling, scaling_count)
    if "gemm_comparison" in experiments:
        gemm_warmup, gemm_iterations = (1, 1) if args.kind == "pilot" else (2, 10)
        gemm = run([sys.executable, "gemm_comparison/gemm_comparison.py",
                    "--warmup-iterations", str(gemm_warmup),
                    "--iterations", str(gemm_iterations)])
        counts["gemm_comparison"] = write_rows(raw / "gemm_comparison.csv", gemm, 20)

    profile = {"state": "NOT_REQUESTED", "captured_count": 0}
    if args.with_ncu:
        # Profile after timing so NCU replay cannot affect measured throughput.
        import ncu_capture
        profile = ncu_capture.capture(directory)

    metadata = {"campaign_id": campaign_id, "kind": args.kind, "created_utc": now.isoformat(),
                "gpu": gpu_info(), "experiments": list(experiments), "row_counts": counts,
                "telemetry": telemetry,
                "ncu": {key: profile[key] for key in ("state", "captured_count")}}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"campaign: COMPLETE {directory}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"campaign: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
