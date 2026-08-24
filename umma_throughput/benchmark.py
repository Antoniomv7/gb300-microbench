#!/usr/bin/env python3
"""Run isolated UMMA sweeps or the four whole-device configurations."""

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("umma_1sm", "umma_2sm")
N_VALUES = (64, 128, 256)
DEPTHS = (4, 16, 64, 256)


def run(command, expected_rows):
    result = subprocess.run(
        [str(ROOT / "scripts/run_gpu.sh"), *command], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, check=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) != expected_rows + 1:
        raise RuntimeError(f"expected {expected_rows} samples, received {len(rows) - 1}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run BF16 UMMA microbenchmarks.")
    parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    parser.add_argument("--device-scaling", action="store_true")
    parser.add_argument("--campaign-kind", default="none", choices=("none", "pilot", "final"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("BLACKWELL_GPU_INDEX"):
        raise SystemExit("set BLACKWELL_GPU_INDEX")

    defaults = (20, 5, 3) if args.run_kind == "smoke" else (1000, 10, 30)
    iterations = args.iterations or defaults[0]
    warmup = defaults[1] if args.warmup_iterations is None else args.warmup_iterations
    repetitions = args.repetitions or defaults[2]
    common = ["--run-kind", args.run_kind, "--iterations", str(iterations),
              "--warmup-iterations", str(warmup), "--repetitions", str(repetitions)]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if args.device_scaling:
            command = ["build/umma_throughput/umma_device_scaling", *common,
                       "--campaign-kind", args.campaign_kind]
            print("umma: isolated versus whole-device scaling", file=sys.stderr)
            rows = run(command, 4 * repetitions)
            correctness = rows[0].index("correctness")
            if any(row[correctness] != "OK" for row in rows[1:]):
                raise RuntimeError("device-scale UMMA failed numerical validation")
            writer.writerows(rows)
            return

        header = None
        for n in N_VALUES:
            for depth in DEPTHS:
                for method in METHODS:
                    print(f"umma: {method}, N={n}, depth={depth}", file=sys.stderr)
                    rows = run([f"build/umma_throughput/{method}", *common,
                                "--n", str(n), "--depth", str(depth)], repetitions)
                    if header is None:
                        header = rows[0]
                        writer.writerow(header)
                    elif rows[0] != header:
                        raise RuntimeError("UMMA benchmarks returned different CSV headers")
                    writer.writerows(rows[1:])
                    handle.flush()


if __name__ == "__main__":
    main()
