#!/usr/bin/env python3
"""Sweep the matched LDGSTS and TMA configurations."""

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ldgsts", "tma")
STAGES = (2, 4, 8)
IN_FLIGHT_KIB = (16, 32, 64)


def run(command, repetitions):
    result = subprocess.run(
        [str(ROOT / "scripts/run_gpu.sh"), *command], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, check=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != repetitions + 1:
        raise RuntimeError(f"expected {repetitions} samples, received {len(rows) - 1}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare LDGSTS and TMA.")
    parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    parser.add_argument("--working-set-mib", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument("--warmup-ms", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("BLACKWELL_GPU_INDEX"):
        raise SystemExit("set BLACKWELL_GPU_INDEX")

    defaults = (64, 2, 200, 5) if args.run_kind == "smoke" else (512, 32, 2000, 30)
    working_set = args.working_set_mib or defaults[0]
    passes = args.passes or defaults[1]
    warmup = defaults[2] if args.warmup_ms is None else args.warmup_ms
    repetitions = args.repetitions or defaults[3]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        header = None
        for stages in STAGES:
            for in_flight in IN_FLIGHT_KIB:
                for method in METHODS:
                    print(f"memory: {method}, stages={stages}, in_flight={in_flight} KiB", file=sys.stderr)
                    rows = run([
                        f"build/memory_paths/{method}", "--stages", str(stages),
                        "--bytes-in-flight-kib", str(in_flight), "--run-kind", args.run_kind,
                        "--working-set-mib", str(working_set), "--passes", str(passes),
                        "--warmup-ms", str(warmup), "--repetitions", str(repetitions),
                    ], repetitions)
                    if header is None:
                        header = rows[0]
                        writer.writerow(header)
                    elif rows[0] != header:
                        raise RuntimeError("memory benchmarks returned different CSV headers")
                    writer.writerows(rows[1:])
                    handle.flush()


if __name__ == "__main__":
    main()
