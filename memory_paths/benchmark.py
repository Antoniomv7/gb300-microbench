#!/usr/bin/env python3
"""Run the complete LDGSTS/TMA configuration grid and write one raw CSV."""

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_gpu.sh"
METHODS = ("ldgsts", "tma")
STAGES = (2, 4, 8)
BYTES_IN_FLIGHT_KIB = (16, 32, 64)


def integer(minimum):
    def parse(text):
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return value
    return parse


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2 methods x 3 stage counts x 3 in-flight sizes.")
    parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    parser.add_argument("--working-set-mib", type=integer(1))
    parser.add_argument("--passes", type=integer(1))
    parser.add_argument("--warmup-ms", type=integer(0))
    parser.add_argument("--repetitions", type=integer(1))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def run_one(command, repetitions):
    completed = subprocess.run(
        [str(LAUNCHER), *command], cwd=ROOT, env=os.environ.copy(),
        text=True, capture_output=True, check=False,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"{' '.join(command)} failed with exit code {completed.returncode}")
    table = list(csv.reader(io.StringIO(completed.stdout)))
    if not table or len(table) != repetitions + 1:
        raise RuntimeError(f"{' '.join(command)} returned {max(len(table)-1, 0)} rows, expected {repetitions}")
    return table[0], table[1:]


def main():
    args = parse_args()
    if "BLACKWELL_GPU_INDEX" not in os.environ:
        raise SystemExit("BLACKWELL_GPU_INDEX must be set explicitly")

    defaults = {
        "smoke": (64, 2, 200, 5),
        "benchmark": (512, 32, 2000, 30),
    }[args.run_kind]
    working_set, passes, warmup, repetitions = (
        args.working_set_mib or defaults[0],
        args.passes or defaults[1],
        defaults[2] if args.warmup_ms is None else args.warmup_ms,
        args.repetitions or defaults[3],
    )

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    header = None
    written = 0
    # Each configuration is flushed as soon as it completes, so a failure in
    # the last one still leaves every earlier configuration on disk.
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for stages in STAGES:
            for bif in BYTES_IN_FLIGHT_KIB:
                for method in METHODS:
                    print(f"memory_paths: {method} stages={stages} bytes_in_flight_kib={bif}", file=sys.stderr)
                    command = [
                        f"build/memory_paths/{method}",
                        "--stages", str(stages),
                        "--bytes-in-flight-kib", str(bif),
                        "--run-kind", args.run_kind,
                        "--working-set-mib", str(working_set),
                        "--passes", str(passes),
                        "--warmup-ms", str(warmup),
                        "--repetitions", str(repetitions),
                    ]
                    current_header, current_rows = run_one(command, repetitions)
                    if header is None:
                        header = current_header
                        writer.writerow(header)
                    elif current_header != header:
                        raise RuntimeError("CSV schema changed between configurations")
                    writer.writerows(current_rows)
                    handle.flush()
                    written += len(current_rows)

    print(f"memory_paths: wrote {written} rows to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

