#!/usr/bin/env python3
"""Capture DRAM counters and the SM clock with Nsight Compute."""

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY_METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
UMMA_METRICS = ("sm__cycles_elapsed.avg.per_second", "gpu__time_duration.sum")

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


def parse_ncu_csv(text, requested):
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("empty NCU export")
    header = [field.strip().lstrip("\ufeff") for field in rows[0]]
    values, units = {}, {}

    def metric_name(value):
        return next((name for name in requested if value == name or value.endswith("." + name)), None)

    if "Metric Name" in header:
        name_column = header.index("Metric Name")
        value_column = header.index("Metric Value")
        unit_column = header.index("Metric Unit") if "Metric Unit" in header else None
        for row in rows[1:]:
            name = metric_name(row[name_column].strip())
            if name:
                values[name] = float(row[value_column].replace(",", ""))
                units[name] = row[unit_column].strip() if unit_column is not None else ""
    else:
        if len(rows) < 3:
            raise ValueError("NCU export has no metric values")
        for index, value in enumerate(header):
            name = metric_name(value)
            if name:
                values[name] = float(rows[-1][index].replace(",", ""))
                units[name] = rows[1][index].strip()

    if any(name not in values or not math.isfinite(values[name]) for name in requested):
        raise ValueError("NCU did not provide every requested counter")
    return values, units


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
    ncu = os.environ.get("NCU_BINARY", "ncu")
    with tempfile.TemporaryDirectory(prefix="gb300-ncu-") as temporary:
        report = Path(temporary) / "report"
        collect = [ncu, "--clock-control", "none", "--pipeline-boost-state", "dynamic",
                   "--cache-control", "none", "--kernel-name-base", "function",
                   "--kernel-name", case["kernel_name"], "--launch-count", "1",
                   "--devices", "0", "--replay-mode", "kernel", "--print-summary", "none",
                   "--metrics", ",".join(case["metrics"]), "-o", str(report), "--",
                   *benchmark_command(case)]
        application = subprocess.run(collect, cwd=ROOT, text=True, capture_output=True,
                                     timeout=900, check=True)
        exported = subprocess.run(
            [ncu, "--csv", "--page", "raw", "--print-units", "base", "--print-fp",
             "--print-kernel-base", "function", "--import", str(report) + ".ncu-rep"],
            cwd=ROOT, text=True, capture_output=True, timeout=900, check=True)

    metrics, units = parse_ncu_csv(exported.stdout, case["metrics"])
    output = directory / f"{case['case']}.csv"
    output.write_text(exported.stdout, encoding="utf-8")
    record = {key: value for key, value in case.items() if key != "metrics"}
    record.update({"status": "captured", "csv": output.name, "metrics": metrics, "units": units})
    if case["section"] == "memory_paths":
        row = next(csv.DictReader(io.StringIO(application.stdout)))
        record["useful_bytes"] = float(row["useful_bytes"])
    return record


def capture(campaign):
    directory = Path(campaign) / "ncu"
    directory.mkdir(exist_ok=False)
    cases = []
    for case in (*MEMORY_PLAN, *UMMA_PLAN):
        print(f"ncu: {case['case']}", file=sys.stderr, flush=True)
        cases.append(capture_case(case, directory))
    index = {"state": "COMPLETE", "captured_count": len(cases), "cases": cases}
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture the essential NCU counters.")
    parser.add_argument("--campaign-dir", required=True, type=Path)
    capture(parser.parse_args().campaign_dir)
