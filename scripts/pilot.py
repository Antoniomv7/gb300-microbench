#!/usr/bin/env python3
"""Compile first with make pilot; exercise all new GPU paths before the final study."""

import argparse
import csv
import io
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import metadata  # noqa: E402


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def run(directory, name, command):
    print(f"pilot: {name}", file=sys.stderr, flush=True)
    with (directory/f"{name}.stderr.log").open("w") as error:
        result = subprocess.run([str(v) for v in command], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=error, check=False)
    (directory/f"{name}.stdout").write_text(result.stdout)
    if result.returncode:
        error_path = directory/f"{name}.stderr.log"
        print(f"pilot: {name} failed; last stderr lines ({error_path}):", file=sys.stderr)
        print("\n".join(error_path.read_text().splitlines()[-40:]), file=sys.stderr, flush=True)
    result.check_returncode()
    return list(csv.DictReader(io.StringIO(result.stdout))) if result.stdout.strip() else []


def verify(directory):
    report = json.loads((directory/"report.json").read_text())
    if report["status"] != "PASS" or not report.get("completed_utc"):
        raise ValueError("a completed passing pilot is required")
    if report["git_commit"] != metadata.source_commit() or report.get("protocol_version") != 2:
        raise ValueError("the pilot belongs to another source commit/protocol")
    if os.environ.get("BLACKWELL_GPU_UUID"):
        current = metadata.environment()
        if current["environment_fingerprint"] != report["environment_fingerprint"]:
            raise ValueError("the GPU/software/power-limit environment changed since the pilot")
    return report


def pilot(directory):
    directory.mkdir(parents=True, exist_ok=False)
    record = {"pilot": True, "created_utc": metadata.utc_now(), **metadata.environment(),
              "status": "RUNNING"}
    (directory/"report.json").write_text(json.dumps(record, indent=2)+"\n")
    isolated_args = ["--run-kind", "smoke", "--n", "256", "--depth", "256",
                     "--iterations", "1000", "--warmup-iterations", "10", "--repetitions", "3"]
    for method in ("umma_1sm", "umma_2sm"):
        rows = run(directory, method, [f"build/umma_throughput/{method}", *isolated_args])
        if len(rows) != 3 or any(row["correctness"] != "OK" or
                not 0 < float(row["timer_bracket_ns"])+1 < float(row["loop_elapsed_ns"])*0.01
                for row in rows):
            raise ValueError("isolated loop-clock pilot failed")
    scaling_args = ["build/umma_throughput/umma_device_scaling", "--run-kind", "smoke",
                    "--iterations", "1000", "--warmup-iterations", "10", "--repetitions", "3"]
    measured = run(directory, "scaling_instrumented", [*scaling_args,
                   "--loop-output", directory/"loops.csv"])
    control = run(directory, "scaling_control", [*scaling_args, "--instrument-loops", "0"])
    if len(measured) != 12 or len(control) != 12:
        raise ValueError("scaling pilot missed a configuration")
    groups = defaultdict(list)
    for row in read_csv(directory/"loops.csv"):
        ns, cycles = float(row["loop_elapsed_ns"]), float(row["elapsed_cycles"])
        if ns <= 0 or cycles <= 0 or float(row["timer_bracket_ns"]) >= ns*0.01 or not math.isclose(
                float(row["effective_sm_clock_mhz"]), 1000*cycles/ns, rel_tol=1e-7):
            raise ValueError("invalid internal SM-clock observation")
        groups[(row["method"], row["scale"], row["sample_index"])].append(row)
    for row in measured:
        samples = groups[(row["method"], row["scale"], row["sample_index"])]
        if len(samples) != int(row["planned_active_sm_count"]) or len(
                {sample["smid"] for sample in samples}) != len(samples):
            raise ValueError("missing or duplicate timed SM observations")
    def medians(rows):
        values = defaultdict(list)
        for row in rows:
            if row["correctness"] != "OK":
                raise ValueError("scaling correctness failed")
            values[(row["method"], row["scale"])].append(float(row["kernel_time_ms"]))
        return {key: statistics.median(v) for key, v in values.items()}
    a, b = medians(measured), medians(control)
    record["instrumented_to_control_time_ratio"] = {"/".join(k): a[k]/b[k] for k in a}
    # These are separate launches under dynamic clocks, not a calibrated subtraction
    # of timestamp latency. Flag a large discrepancy for inspection in the pilot.
    if any(not 0.8 <= value <= 1.2 for value in record["instrumented_to_control_time_ratio"].values()):
        raise ValueError("instrumentation/control differs by more than 20%; inspect before acquisition")
    gemm = run(directory, "gemm", [sys.executable, "gemm_comparison/gemm_comparison.py",
               "--pilot", "--warmup-iterations", "5", "--iterations", "20"])
    if len(gemm) != 5 or any(row["correctness"] != "PASS" or row["timing_method"] != "cuda_graph"
                            for row in gemm):
        raise ValueError("a GEMM variant or graph timing path failed")
    record["gemm_host_to_graph_ratios"] = {r["variant"]: float(r["host_to_graph_time_ratio"]) for r in gemm}
    run(directory, "precision", [sys.executable, "precision_comparison/precision_comparison.py",
                                "--pilot", "--output", directory/"precision"])
    run(directory, "power", [sys.executable, "scripts/power_measure.py",
                            "--pilot", "--output", directory/"power"])
    run(directory, "profile", [sys.executable, "scripts/profile_gemm.py",
                              "--pilot", "--cache-state", "hot", "--output", directory/"profile"])
    power = json.loads((directory/"power/metadata.json").read_text())
    record["short_power_windows_stationary"] = power["all_windows_stationary"]
    record["power_note"] = "3-second functional probe; final acquisition enforces 10s warm-up/30s windows"
    record["status"] = "PASS"
    metadata.complete(record)
    (directory/"report.json").write_text(json.dumps(record, indent=2)+"\n")
    print(f"pilot: PASS; archive and review {directory}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print("pilot: source/protocol gate passed")
    else:
        pilot(args.output.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"pilot: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
