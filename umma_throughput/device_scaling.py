#!/usr/bin/env python3
"""Run the four isolated/device-scale UMMA configurations and write one raw CSV.

The binary owns the interleaving: it measures ``umma_1sm``/``umma_2sm`` at both
launch scales inside a single process, alternating their order between
repetitions, so this driver invokes it exactly once and only checks that the
returned table matches the supplementary contract.  It never appends to the
frozen ``umma_throughput.csv``.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_gpu.sh"
BINARY = "build/umma_throughput/umma_device_scaling"

CONFIGURATIONS = (
    ("umma_1sm", "isolated"),
    ("umma_2sm", "isolated"),
    ("umma_1sm", "device_scale"),
    ("umma_2sm", "device_scale"),
)

# (iterations, warmup_iterations, repetitions)
DEFAULTS = {"smoke": (20, 5, 3), "benchmark": (1000, 10, 30)}


def integer(minimum):
    def parse(text):
        value = int(text)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return value
    return parse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 2 UMMA methods x 2 launch scales at N=256, depth=256."
    )
    parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    parser.add_argument("--campaign-kind", default="none", choices=("none", "pilot", "final"))
    parser.add_argument("--iterations", type=integer(1))
    parser.add_argument("--warmup-iterations", type=integer(0))
    parser.add_argument("--repetitions", type=integer(1))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "BLACKWELL_GPU_INDEX" not in os.environ:
        raise SystemExit("BLACKWELL_GPU_INDEX must be set explicitly")

    defaults = DEFAULTS[args.run_kind]
    iterations = args.iterations or defaults[0]
    warmup = defaults[1] if args.warmup_iterations is None else args.warmup_iterations
    repetitions = args.repetitions or defaults[2]

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        BINARY,
        "--run-kind", args.run_kind,
        "--campaign-kind", args.campaign_kind,
        "--iterations", str(iterations),
        "--warmup-iterations", str(warmup),
        "--repetitions", str(repetitions),
    ]
    print(f"umma_device_scaling: {' '.join(command)}", file=sys.stderr)
    completed = subprocess.run(
        [str(LAUNCHER), *command], cwd=ROOT, env=os.environ.copy(),
        text=True, capture_output=True, check=False,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode:
        raise SystemExit(f"{BINARY} failed with exit code {completed.returncode}")

    table = list(csv.reader(io.StringIO(completed.stdout)))
    expected_rows = len(CONFIGURATIONS) * repetitions
    if len(table) != expected_rows + 1:
        raise SystemExit(
            f"{BINARY} returned {max(len(table) - 1, 0)} rows, expected {expected_rows}"
        )
    header = table[0]
    index = {name: position for position, name in enumerate(header)}
    for required in ("method", "scale", "sample_index", "execution_order", "correctness"):
        if required not in index:
            raise SystemExit(f"{BINARY} output is missing the {required!r} column")
    seen: dict[tuple[str, str], set[int]] = {key: set() for key in CONFIGURATIONS}
    for number, row in enumerate(table[1:], 2):
        key = (row[index["method"]], row[index["scale"]])
        if key not in seen:
            raise SystemExit(f"row {number}: unexpected configuration {key}")
        sample = int(row[index["sample_index"]])
        if sample in seen[key]:
            raise SystemExit(f"row {number}: duplicate sample {key + (sample,)}")
        seen[key].add(sample)
        if row[index["correctness"]] != "OK":
            raise SystemExit(f"row {number}: correctness is not OK")
    for key, samples in seen.items():
        if samples != set(range(repetitions)):
            raise SystemExit(f"{key}: sample indexes are not exactly 0..{repetitions - 1}")
    orders = sorted(int(row[index["execution_order"]]) for row in table[1:])
    if orders != list(range(expected_rows)):
        raise SystemExit(f"execution_order is not exactly 0..{expected_rows - 1}")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(table)

    print(f"umma_device_scaling: wrote {expected_rows} rows to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
