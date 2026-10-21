#!/usr/bin/env python3
"""Sustained operating power of the two full-device UMMA configurations."""

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import gpu_telemetry, metadata  # noqa: E402

REPETITIONS = 3
WARMUP_MS, DURATION_MS = 10000, 30000
MAX_POWER_DRIFT, MAX_CLOCK_DRIFT = 0.05, 0.02


def weighted_mean(samples, start, end, field):
    """Integrate linear interpolation, clipped exactly to the two window edges."""
    points = sorted({float(r["unix_s"]): float(r[field]) for r in samples
                     if r.get(field) not in (None, "")}.items())
    if end <= start or len(points) < 2 or points[0][0] > start or points[-1][0] < end:
        raise ValueError(f"{field}: samples do not bracket the measurement window")
    integral, covered = 0.0, 0.0
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t1 <= start or t0 >= end:
            continue
        if not all(math.isfinite(v) for v in (t0, t1, v0, v1)) or t1-t0 > 0.5:
            raise ValueError(f"{field}: invalid value or telemetry gap above 0.5 s")
        a, b = max(start, t0), min(end, t1)
        va, vb = v0+(v1-v0)*(a-t0)/(t1-t0), v0+(v1-v0)*(b-t0)/(t1-t0)
        integral += (va+vb)*(b-a)/2
        covered += b-a
    if not math.isclose(covered, end-start, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("incomplete telemetry integration")
    return integral/(end-start)


def window_summary(samples, start, end):
    mean = lambda field, a=start, b=end: weighted_mean(samples, a, b, field)
    middle = (start+end)/2
    def drift(field):
        a, b = mean(field, start, middle), mean(field, middle, end)
        return abs(b-a)/((a+b)/2) if a+b else 0.0
    power = mean("power_average_w")
    if power <= 0:
        raise ValueError("sustained power must be positive")
    pd, cd = drift("power_average_w"), drift("sm_clock_mhz")
    instant = all(r.get("power_instant_w") not in (None, "") for r in samples)
    return {"mean_power_w": power,
            "mean_instant_power_w": mean("power_instant_w") if instant else "",
            "mean_sm_clock_mhz": mean("sm_clock_mhz"),
            "mean_temperature_c": mean("temperature_c"),
            "power_half_window_relative_change": pd,
            "clock_half_window_relative_change": cd,
            "stationary": pd <= MAX_POWER_DRIFT and cd <= MAX_CLOCK_DRIFT}


def measure(output, pilot=False):
    directory = output if output.is_absolute() else ROOT/output
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory/"raw"
    raw.mkdir()
    warmup, duration = (2000, 3000) if pilot else (WARMUP_MS, DURATION_MS)
    record = {"experiment": "sustained_umma_power", "created_utc": metadata.utc_now(),
              **metadata.environment(), "pilot": pilot,
              "parameters": {"repetitions": 1 if pilot else REPETITIONS,
                             "warmup_ms": warmup, "duration_ms": duration,
                             "power_half_window_limit": MAX_POWER_DRIFT,
                             "clock_half_window_limit": MAX_CLOCK_DRIFT,
                             "poll_interval_ms": 50}}
    rows, sensors = [], []
    record["status"] = "RUNNING"
    (directory/"metadata.json").write_text(json.dumps(record, indent=2)+"\n")
    gpu = os.environ.get("BLACKWELL_GPU_UUID", "0")
    for repetition in range(1, (1 if pilot else REPETITIONS)+1):
        methods = ("umma_1sm", "umma_2sm") if repetition % 2 else ("umma_2sm", "umma_1sm")
        for order, method in enumerate(methods):
            stem = f"r{repetition}_{method}"
            command = ["build/umma_throughput/umma_device_scaling", "--run-kind", "sustained",
                       "--iterations", "1000", "--warmup-iterations", "0", "--repetitions", "1",
                       "--only-method", method, "--only-scale", "device_scale",
                       "--warmup-ms", str(warmup), "--duration-ms", str(duration)]
            print(f"power: {stem}", file=sys.stderr, flush=True)
            sampler = gpu_telemetry.ClockSampler(gpu, 50)
            try:
                with sampler:
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                            timeout=(warmup+duration)/1000+120, check=False)
                    (raw/f"{stem}.stdout.csv").write_text(result.stdout)
                    (raw/f"{stem}.stderr.log").write_text(result.stderr)
                    result.check_returncode()
                    parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                    if len(parsed) != 1 or parsed[0]["correctness"] != "OK":
                        raise RuntimeError("sustained UMMA validation failed")
                    row = parsed[0]
                    start, end = float(row["host_start_unix_s"]), float(row["host_end_unix_s"])
                    sampler.cover_through(end)
            finally:
                sampler.write(raw/f"{stem}.telemetry.csv")
            sensors.append({"case": stem, **sampler.verify()})
            with (raw/f"{stem}.telemetry.csv").open() as source:
                summary = window_summary(list(csv.DictReader(source)), start, end)
            if end-start < duration/1000-0.001 or float(row["gpu_duty_fraction"]) < 0.90:
                raise RuntimeError("sustained window too short or GPU duty fraction below 90%")
            rows.append({"repetition": repetition, "order_in_repetition": order, **row, **summary})
            with (raw/"sustained.csv").open("w", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    record["sensors"] = sensors
    record["all_windows_stationary"] = all(row["stationary"] for row in rows)
    (directory/"metadata.json").write_text(json.dumps(record, indent=2)+"\n")
    if not pilot and not record["all_windows_stationary"]:
        raise RuntimeError("power/clock drift exceeds the declared limit; raw windows retained; "
                           "inspect before changing the protocol")
    record["status"] = "COMPLETE"
    metadata.complete(record)
    (directory/"metadata.json").write_text(json.dumps(record, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    measure(args.output, args.pilot)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"power: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
