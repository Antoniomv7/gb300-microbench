#!/usr/bin/env python3
"""Capture DRAM counters and the SM clock with Nsight Compute, and read its raw-page exports."""

import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY_METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
UMMA_METRICS = ("sm__cycles_elapsed.avg.per_second", "gpu__time_duration.sum")
EXPORT_TIMEOUT_S = 900

# Profile representative memory cases and both peak-depth UMMA variants.
MEMORY_PLAN = tuple(
    {"section": "memory_paths", "case": f"{index:02d}_{method}_s{stages}_bif{size}",
     "method": method, "stages": stages, "bytes_in_flight_kib": size,
     "kernel_name": f"{method}_benchmark_kernel", "metrics": MEMORY_METRICS}
    for index, (method, stages, size) in enumerate((
        ("ldgsts", 2, 16), ("tma", 2, 16), ("tma", 4, 32),
        ("ldgsts", 4, 32), ("ldgsts", 8, 64), ("tma", 8, 64)))
)
UMMA_PLAN = tuple(
    {"section": "umma_throughput", "case": f"{index:02d}_{method}_n256_d256",
     "method": method, "n": 256, "depth": 256,
     "kernel_name": f"{method}_m{128 if method == 'umma_1sm' else 256}n256k16_d256",
     "metrics": UMMA_METRICS}
    for index, method in enumerate(("umma_1sm", "umma_2sm"))
)
PLAN = (*MEMORY_PLAN, *UMMA_PLAN)


def planned_cases(experiments):
    """The captures that belong to the selected experiments; a full campaign has all eight."""
    return tuple(case for case in PLAN if case["section"] in experiments)


def ncu(*arguments):
    return [os.environ.get("NCU_BINARY", "ncu"), *map(str, arguments)]


def export_csv(report, destination):
    """Export a report's raw page in base units and keep the export beside the report."""
    completed = subprocess.run(
        ncu("--import", report, "--csv", "--page", "raw", "--print-units", "base",
            "--print-fp", "--print-kernel-base", "function"),
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=EXPORT_TIMEOUT_S)
    Path(destination).write_text(completed.stdout, encoding="utf-8")
    return completed.stdout


def parse_kernels(text, metrics):
    """Return one record per profiled kernel from a raw-page NCU CSV export, and the units."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("empty NCU export")
    header = [field.strip().lstrip("﻿") for field in rows[0]]
    missing = [name for name in ("Kernel Name", *metrics) if name not in header]
    if missing:
        raise ValueError(f"NCU export lacks {missing}")
    units = {metric: rows[1][header.index(metric)].strip() for metric in metrics}
    nvtx = next((index for index, name in enumerate(header) if "Push/Pop_Range" in name), None)
    kernels = []
    for row in rows[2:]:
        column = dict(zip(header, row))
        kernels.append({
            "name": column["Kernel Name"], "block_size": column.get("Block Size", ""),
            "grid_size": column.get("Grid Size", ""),
            "nvtx_ranges": row[nvtx].strip() if nvtx is not None else "",
            "metrics": {metric: float(column[metric].replace(",", "")) for metric in metrics}})
    return kernels, units


def benchmark_command(case):
    if case["section"] == "memory_paths":
        return [f"build/memory_paths/{case['method']}", "--stages", str(case["stages"]),
                "--bytes-in-flight-kib", str(case["bytes_in_flight_kib"]), "--run-kind",
                "benchmark", "--working-set-mib", "512", "--passes", "32",
                "--warmup-ms", "0", "--repetitions", "1"]
    return [f"build/umma_throughput/{case['method']}", "--run-kind", "benchmark",
            "--n", "256", "--depth", "256", "--iterations", "1000",
            "--warmup-iterations", "0", "--repetitions", "1"]


def capture_case(case, directory):
    report = directory / case["case"]
    collect = ncu("--clock-control", "none", "--pipeline-boost-state", "dynamic",
                  "--cache-control", "none", "--kernel-name-base", "function",
                  "--kernel-name", case["kernel_name"], "--launch-count", "1",
                  "--devices", "0", "--replay-mode", "kernel", "--print-summary", "none",
                  "--metrics", ",".join(case["metrics"]), "-o", report, "--",
                  *benchmark_command(case))
    application = subprocess.run(collect, cwd=ROOT, text=True, capture_output=True,
                                 timeout=900, check=True)
    kernels, units = parse_kernels(export_csv(f"{report}.ncu-rep", f"{report}.csv"),
                                   case["metrics"])
    if len(kernels) != 1 or not all(math.isfinite(value)
                                    for value in kernels[0]["metrics"].values()):
        raise ValueError(f"{case['case']}: NCU did not provide every requested counter once")
    record = {key: value for key, value in case.items() if key != "metrics"}
    record.update({"status": "captured", "report": f"{case['case']}.ncu-rep",
                   "csv": f"{case['case']}.csv", "metrics": kernels[0]["metrics"],
                   "units": units})
    if case["section"] == "memory_paths":
        # Profiler messages may appear before the benchmark CSV header.
        lines = application.stdout.splitlines()
        header = next(index for index, line in enumerate(lines) if line.startswith("method,"))
        row = next(csv.DictReader(lines[header:]))
        record["useful_bytes"] = float(row["useful_bytes"])
    return record


def capture(campaign, cases):
    directory = Path(campaign) / "ncu"
    directory.mkdir(exist_ok=False)
    records = []
    for case in cases:
        print(f"ncu: {case['case']}", file=sys.stderr, flush=True)
        records.append(capture_case(case, directory))
    index = {"state": "COMPLETE", "captured_count": len(records), "cases": records}
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index
