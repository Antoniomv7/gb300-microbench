#!/usr/bin/env python3
"""Capture the NCU counters needed for DRAM checks and SM frequency."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_gpu.sh"
MEMORY_METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
UMMA_METRICS = ("sm__cycles_elapsed.avg.per_second", "gpu__time_duration.sum")

MEMORY_PLAN = tuple(
    {
        "section": "memory_paths", "case": f"{index:02d}_{method}_s{stages}_bif{size}",
        "method": method, "stages": stages, "bytes_in_flight_kib": size,
        "kernel_name": f"{method}_benchmark_kernel", "metrics": MEMORY_METRICS,
    }
    for index, (method, stages, size) in enumerate((
        ("ldgsts", 2, 16), ("tma", 2, 16), ("tma", 4, 32),
        ("ldgsts", 4, 32), ("ldgsts", 8, 64), ("tma", 8, 64),
    ))
)
UMMA_PLAN = tuple(
    {
        "section": "umma_throughput", "case": f"{index:02d}_{method}_n256_d256",
        "method": method, "n": 256, "depth": 256,
        "kernel_name": f"{method}_m{128 if method == 'umma_1sm' else 256}n256k16_d256",
        "metrics": UMMA_METRICS,
    }
    for index, method in enumerate(("umma_1sm", "umma_2sm"))
)


def parse_ncu_raw_csv(text, candidates):
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("empty NCU export")
    header = [field.strip().lstrip("\ufeff") for field in rows[0]]
    result, units = {}, {}

    def requested(name):
        return next((item for item in candidates if name == item or name.endswith("." + item)), None)

    if "Metric Name" in header:
        name_index, value_index = header.index("Metric Name"), header.index("Metric Value")
        unit_index = header.index("Metric Unit") if "Metric Unit" in header else None
        for row in rows[1:]:
            metric = requested(row[name_index].strip())
            if metric:
                try:
                    value = float(row[value_index].replace(",", ""))
                except ValueError:
                    continue
                if math.isfinite(value):
                    result[metric] = value
                    units[metric] = row[unit_index].strip() if unit_index is not None else ""
    else:
        if len(rows) < 3:
            raise ValueError("NCU export is missing its values")
        values = rows[-1]
        for index, name in enumerate(header):
            metric = requested(name)
            if metric:
                try:
                    value = float(values[index].replace(",", ""))
                except (ValueError, IndexError):
                    continue
                if math.isfinite(value):
                    result[metric] = value
                    units[metric] = rows[1][index].strip()
    if not result:
        raise ValueError("no requested NCU counter was available")
    return {"metrics": result, "units": units}


def benchmark_argv(entry):
    if entry["section"] == "memory_paths":
        return [f"build/memory_paths/{entry['method']}", "--stages", str(entry["stages"]),
                "--bytes-in-flight-kib", str(entry["bytes_in_flight_kib"]),
                "--run-kind", "benchmark", "--working-set-mib", "512", "--passes", "32",
                "--warmup-ms", "0", "--repetitions", "1"]
    return [f"build/umma_throughput/{entry['method']}", "--run-kind", "benchmark",
            "--n", "256", "--depth", "256", "--iterations", "1000",
            "--warmup-iterations", "0", "--repetitions", "1"]


def useful_bytes(application_output):
    try:
        row = next(csv.DictReader(io.StringIO(application_output)))
        value = float(row["useful_bytes"])
    except (KeyError, StopIteration, ValueError):
        return None
    return value if value > 0 else None


def capture_case(entry, directory, ncu_binary):
    record = {key: value for key, value in entry.items() if key != "metrics"}
    record.update({"status": "unavailable", "metrics": {}})
    output = directory / f"{entry['case']}.csv"
    container_output = "/workspace/" + str(output.relative_to(ROOT))
    collect = [ncu_binary, "--clock-control", "none", "--pipeline-boost-state", "dynamic",
               "--cache-control", "none", "--kernel-name-base", "function",
               "--kernel-name", entry["kernel_name"], "--launch-count", "1", "--devices", "0",
               "--replay-mode", "kernel", "--print-summary", "none", "--metrics",
               ",".join(entry["metrics"])]
    export = [ncu_binary, "--csv", "--page", "raw", "--print-units", "base",
              "--print-fp", "--print-kernel-base", "function"]
    program = (
        'set -eu\nwork="$(mktemp -d)"\ntrap \'rm -rf "$work"\' EXIT\n'
        + shlex.join(collect) + ' -o "$work/report" -- '
        + shlex.join(benchmark_argv(entry)) + ' > "$work/application.csv"\n'
        + shlex.join(export) + ' --import "$work/report.ncu-rep" > '
        + shlex.quote(container_output) + '\ncat "$work/application.csv"\n'
    )
    try:
        completed = subprocess.run([str(LAUNCHER), "bash", "-c", program], cwd=ROOT,
                                   text=True, capture_output=True, timeout=900, check=True)
        parsed = parse_ncu_raw_csv(output.read_text(encoding="utf-8"), entry["metrics"])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record["reason"] = str(error)
        output.unlink(missing_ok=True)
        return record
    record.update({"status": "captured", "csv": output.name, **parsed})
    if entry["section"] == "memory_paths":
        record["useful_bytes"] = useful_bytes(completed.stdout)
    return record


def capture(campaign_dir, case_name=None):
    campaign_dir = Path(campaign_dir).resolve()
    if not campaign_dir.is_dir() or not campaign_dir.is_relative_to(ROOT):
        raise RuntimeError("NCU needs an existing campaign inside the repository")
    output = campaign_dir / "ncu"
    output.mkdir(exist_ok=False)
    entries = [entry for entry in (*MEMORY_PLAN, *UMMA_PLAN)
               if case_name is None or entry["case"] == case_name]
    if not entries:
        raise RuntimeError(f"unknown NCU case: {case_name}")
    cases = []
    for entry in entries:
        print(f"ncu: {entry['case']}", file=sys.stderr)
        record = capture_case(entry, output, os.environ.get("NCU_BINARY", "ncu"))
        if record["status"] != "captured":
            print(f"ncu: unavailable ({record.get('reason', 'unknown')})", file=sys.stderr)
        cases.append(record)
    captured = sum(item["status"] == "captured" for item in cases)
    state = "COMPLETE" if captured == len(cases) else "PARTIAL" if captured else "UNAVAILABLE"
    index = {"state": state, "case_count": len(cases), "captured_count": captured, "cases": cases}
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"ncu: {captured}/{len(cases)} cases captured", file=sys.stderr)
    return index


def main():
    parser = argparse.ArgumentParser(description="Capture the essential Nsight Compute counters.")
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--case")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    index = capture(args.campaign_dir, args.case)
    if args.require_complete and index["state"] != "COMPLETE":
        raise SystemExit("NCU did not capture every requested counter")


if __name__ == "__main__":
    main()
