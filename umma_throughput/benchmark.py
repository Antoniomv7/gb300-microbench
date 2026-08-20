#!/usr/bin/env python3
"""Run the complete 1-SM/2-SM UMMA grid and write one raw CSV."""

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_gpu.sh"
METHODS = ("umma_1sm", "umma_2sm")
N_VALUES = (64, 128, 256)
DEPTHS = (4, 16, 64, 256)


def integer(minimum):
    def parse(text):
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return value
    return parse


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2 methods x 3 N values x 4 depths.")
    parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    parser.add_argument("--iterations", type=integer(1))
    parser.add_argument("--warmup-iterations", type=integer(0))
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
        "smoke": (20, 5, 3),
        "benchmark": (1000, 10, 30),
    }[args.run_kind]
    iterations = args.iterations or defaults[0]
    warmup = defaults[1] if args.warmup_iterations is None else args.warmup_iterations
    repetitions = args.repetitions or defaults[2]

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    header = None
    written = 0
    # Each configuration is flushed as soon as it completes, so a failure in
    # the last one still leaves every earlier configuration on disk.
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for n in N_VALUES:
            for depth in DEPTHS:
                for method in METHODS:
                    print(f"umma_throughput: {method} n={n} depth={depth}", file=sys.stderr)
                    command = [
                        f"build/umma_throughput/{method}",
                        "--run-kind", args.run_kind,
                        "--n", str(n),
                        "--depth", str(depth),
                        "--iterations", str(iterations),
                        "--warmup-iterations", str(warmup),
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

    print(f"umma_throughput: wrote {written} rows to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

