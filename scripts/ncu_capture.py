#!/usr/bin/env python3
"""Collect Nsight Compute counters for a subset of one campaign's kernels.

The stage is optional and advisory.  It profiles the six memory
configurations that anchor the HBM check and all twenty-four UMMA
configurations that carry the SM clock, exports each report with
``ncu --page raw --csv`` and stores the exported CSV, unmodified, inside the
campaign directory.  A counter-permission refusal, a missing profiler or any
other capture failure is recorded as a reason and never stops the campaign:
``analysis/analyze.py`` reports whatever was captured and marks the rest as
unavailable.

Nsight Compute only ever sees paths inside a private directory in the
container's own ``/tmp``; the exported bytes travel back over stdout.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import datetime as dt
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_gpu.sh"

DIRECTORY = "ncu"
INDEX_NAME = "index.json"
INDEX_SCHEMA = "gb300.ncu.v1"

# The six memory configurations that carry the DRAM cross-check: one
# bytes-in-flight size per stage count, for both methods.
MEMORY_PLAN = tuple(
    {
        "section": "memory_paths",
        "case": f"{index:02d}_{method}_s{stages}_bif{bif_kib}",
        "method": method,
        "stages": stages,
        "bytes_in_flight_kib": bif_kib,
        "kernel_name": f"{method}_benchmark_kernel",
        "binary": f"build/memory_paths/{method}",
    }
    for index, (method, stages, bif_kib) in enumerate(
        (
            ("ldgsts", 2, 16), ("tma", 2, 16), ("tma", 4, 32),
            ("ldgsts", 4, 32), ("ldgsts", 8, 64), ("tma", 8, 64),
        )
    )
)

UMMA_PLAN = tuple(
    {
        "section": "umma_throughput",
        "case": f"{index:02d}_{method}_n{n}_d{depth}",
        "method": method,
        "n": n,
        "depth": depth,
        "kernel_name": (
            f"umma_1sm_m128n{n}k16_d{depth}" if method == "umma_1sm"
            else f"umma_2sm_m256n{n}k16_d{depth}"
        ),
        "binary": f"build/umma_throughput/{method}",
    }
    for index, (method, n, depth) in enumerate(
        (method, n, depth)
        for n in (64, 128, 256)
        for depth in (4, 16, 64, 256)
        for method in ("umma_1sm", "umma_2sm")
    )
)

# Memory: the DRAM read counter is the one the HBM check needs; the rest are
# recorded as context.  UMMA: the SM frequency is the one the per-SM ceiling
# needs, the tensor-pipe counters are context.
MEMORY_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "gpu__time_duration.sum",
)
UMMA_METRICS = (
    "sm__cycles_elapsed.avg.per_second",
    "gpu__time_duration.sum",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_tensor.sum",
)

COLLECTION_FLAGS = (
    "--clock-control", "none",
    "--pipeline-boost-state", "dynamic",
    "--cache-control", "none",
    "--kernel-name-base", "function",
    "--launch-count", "1",
    "--devices", "0",
    "--replay-mode", "kernel",
    "--print-summary", "none",
)
# The raw page is row-oriented: each metric is emitted as Metric Name,
# Metric Unit and Metric Value.  It already uses metric identifiers;
# --print-metric-name is only valid for the details page.  Request base units
# and floating-point values that Python can parse without display formatting.
EXPORT_FLAGS = (
    "--csv", "--page", "raw",
    "--print-units", "base",
    "--print-fp",
    "--print-kernel-base", "function",
)

NCU_IDENTITY_COLUMNS = ("ID", "Kernel Name")
NCU_METRIC_COLUMNS = ("Metric Name", "Metric Unit", "Metric Value")

PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                      "insufficient permissions")
SEGMENTS = ("app", "err", "log", "metrics", "export_err")
CAPTURE_TIMEOUT_SECONDS = 900


class CaptureError(RuntimeError):
    pass


class NcuParseError(ValueError):
    pass


def canonical_metric(metric_name: str, candidates: tuple[str, ...]) -> str | None:
    """Resolve an exact or namespace-qualified NCU metric identifier."""
    if metric_name in candidates:
        return metric_name
    matches = [name for name in candidates if metric_name.endswith(f".{name}")]
    return matches[0] if len(matches) == 1 else None


def parse_ncu_raw_csv(text: str, candidates: tuple[str, ...]) -> dict:
    """Parse one row-oriented ``ncu --page raw --csv`` export.

    Nsight Compute emits one row per metric.  All rows must describe exactly
    one profiled launch, and at least one requested metric must resolve to a
    finite, non-negative numeric value.
    """
    try:
        table = [
            row for row in csv.reader(text.splitlines(), strict=True)
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as exc:
        raise NcuParseError(f"malformed profiler CSV: {exc}") from exc
    if len(table) < 2:
        raise NcuParseError(
            f"expected a header and at least one profiler row, got {len(table)}")

    header = [cell.strip() for cell in table[0]]
    if header:
        header[0] = header[0].lstrip("\ufeff")
    if len(header) != len(set(header)):
        raise NcuParseError("duplicate column name in the profiler header")
    required = (*NCU_IDENTITY_COLUMNS, *NCU_METRIC_COLUMNS)
    missing = [column for column in required if column not in header]
    if missing:
        raise NcuParseError(f"header lacks {missing}")

    positions = {column: header.index(column) for column in required}
    launches: set[tuple[str, str]] = set()
    metrics: dict[str, float] = {}
    units: dict[str, str] = {}
    for row_number, row in enumerate(table[1:], 2):
        if len(row) != len(header):
            raise NcuParseError(
                f"profiler row {row_number} has width {len(row)}, expected {len(header)}")
        launch_id = row[positions["ID"]].strip()
        kernel_name = row[positions["Kernel Name"]].strip()
        if not launch_id or not kernel_name:
            raise NcuParseError(f"profiler row {row_number} lacks launch identity")
        launches.add((launch_id, kernel_name))

        metric_name = canonical_metric(
            row[positions["Metric Name"]].strip(), candidates)
        if metric_name is None:
            continue
        if metric_name in metrics:
            raise NcuParseError(f"duplicate requested metric {metric_name!r}")
        raw_value = row[positions["Metric Value"]].strip().replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        metrics[metric_name] = value
        units[metric_name] = row[positions["Metric Unit"]].strip()

    if len(launches) != 1:
        raise NcuParseError(
            f"expected exactly one profiled launch, got {len(launches)}")
    if not metrics:
        raise NcuParseError("none of the requested metrics resolved to a usable value")
    _, kernel_name = next(iter(launches))
    return {"kernel_name": kernel_name, "metrics": metrics, "units": units}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def benchmark_argv(entry: dict) -> list[str]:
    """One profiled launch reproduces the campaign's own parameters, so the
    counters describe the configuration the campaign actually measured."""
    if entry["section"] == "memory_paths":
        return [
            entry["binary"],
            "--stages", str(entry["stages"]),
            "--bytes-in-flight-kib", str(entry["bytes_in_flight_kib"]),
            "--run-kind", "benchmark",
            "--working-set-mib", "512",
            "--passes", "32",
            "--warmup-ms", "0",
            "--repetitions", "1",
        ]
    return [
        entry["binary"],
        "--run-kind", "benchmark",
        "--n", str(entry["n"]),
        "--depth", str(entry["depth"]),
        "--iterations", "1000",
        "--warmup-iterations", "0",
        "--repetitions", "1",
    ]


def shell_program(entry: dict, metrics: tuple[str, ...], ncu_binary: str) -> str:
    """Both profiler invocations run in one container so they can share a
    private directory; every path handed to Nsight Compute lives inside it."""
    collect = " ".join(shlex.quote(token) for token in (
        ncu_binary, *COLLECTION_FLAGS,
        "--kernel-name", entry["kernel_name"],
        "--metrics", ",".join(metrics),
    ))
    application = " ".join(shlex.quote(token) for token in benchmark_argv(entry))
    export = " ".join(shlex.quote(token) for token in (ncu_binary, *EXPORT_FLAGS))
    return (
        'set -u\n'
        'work="$(mktemp -d /tmp/ncu_capture.XXXXXX)" || exit 70\n'
        'trap \'rm -rf "$work"\' EXIT\n'
        'status=0\n'
        f'{collect} --log-file "$work/tool.log" -o "$work/report" -- {application}'
        ' >"$work/app.out" 2>"$work/app.err" || status=$?\n'
        'if [ "$status" -eq 0 ]; then\n'
        f'  {export} --import "$work/report.ncu-rep"'
        ' >"$work/metrics.csv" 2>"$work/export.err" || status=$?\n'
        'fi\n'
        'emit() { printf "%s %s\\n" "$1" "$(base64 -w0 "$2" 2>/dev/null || true)"; }\n'
        'printf "status %s\\n" "$status"\n'
        'emit app "$work/app.out"\n'
        'emit err "$work/app.err"\n'
        'emit log "$work/tool.log"\n'
        'emit metrics "$work/metrics.csv"\n'
        'emit export_err "$work/export.err"\n'
    )


def decode_segments(stdout: str) -> dict[str, str]:
    """Reads the ``<name> <base64>`` stream the container program emits."""
    result: dict[str, str] = {}
    for line in stdout.splitlines():
        name, _, payload = line.partition(" ")
        if name == "status":
            result["status"] = payload.strip()
            continue
        if name not in SEGMENTS:
            continue
        try:
            result[name] = base64.b64decode(payload.strip(), validate=True).decode(
                "utf-8", errors="replace")
        except (binascii.Error, ValueError):
            result[name] = ""
    return result


def failure_reason(segments: dict[str, str], status: str) -> str:
    diagnostics = "\n".join(segments.get(name, "") for name in ("err", "log", "export_err"))
    if any(marker in diagnostics for marker in PERMISSION_MARKERS):
        return "counter_permission_denied"
    if "command not found" in diagnostics or "No such file or directory" in diagnostics:
        return "profiler_or_binary_unavailable"
    return f"profiler_exit_{status or 'unknown'}"


def useful_bytes_of(application_csv: str) -> float | None:
    """The profiled run prints its own CSV row; the logical byte count in it
    is what the DRAM ratio is measured against."""
    lines = [line for line in application_csv.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split(",")
    if "useful_bytes" not in header:
        return None
    cells = lines[1].split(",")
    if len(cells) != len(header):
        return None
    try:
        value = float(cells[header.index("useful_bytes")])
    except ValueError:
        return None
    return value if value > 0 else None


def capture_case(entry: dict, *, ncu_binary: str, dry_run: bool) -> tuple[dict, str]:
    """Returns (index entry, exported CSV text).  Never raises for a capture
    failure: the reason is recorded on the returned entry instead."""
    record = {key: value for key, value in entry.items() if key not in ("binary", "metrics")}
    record.update({"status": "unavailable", "reason": "", "csv": "", "useful_bytes": None})
    if dry_run:
        record["reason"] = "dry_run"
        return record, ""

    program = shell_program(entry, entry["metrics"], ncu_binary)
    try:
        completed = subprocess.run(
            [str(LAUNCHER), "bash", "-c", program],
            cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True,
            check=False, timeout=CAPTURE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        record["reason"] = "capture_timeout"
        return record, ""
    except OSError as exc:
        record["reason"] = f"launcher_unavailable:{exc.errno}"
        return record, ""

    if completed.returncode:
        marker = completed.stderr or ""
        record["reason"] = ("counter_permission_denied"
                            if any(m in marker for m in PERMISSION_MARKERS)
                            else f"launcher_exit_{completed.returncode}")
        return record, ""

    segments = decode_segments(completed.stdout)
    status = segments.get("status", "")
    exported = segments.get("metrics", "")
    if status != "0" or not exported.strip():
        record["reason"] = failure_reason(segments, status)
        return record, ""
    try:
        parse_ncu_raw_csv(exported, entry["metrics"])
    except NcuParseError:
        record["reason"] = "profiler_csv_unusable"
        return record, ""

    record["status"] = "captured"
    record["csv"] = f"{entry['case']}.csv"
    if entry["section"] == "memory_paths":
        record["useful_bytes"] = useful_bytes_of(segments.get("app", ""))
    return record, exported


def plan_for(sections: tuple[str, ...], *, case_name: str | None = None) -> list[dict]:
    plan: list[dict] = []
    if "memory_paths" in sections:
        plan.extend(dict(entry, metrics=MEMORY_METRICS) for entry in MEMORY_PLAN)
    if "umma_throughput" in sections:
        plan.extend(dict(entry, metrics=UMMA_METRICS) for entry in UMMA_PLAN)
    if case_name is not None:
        plan = [entry for entry in plan if entry["case"] == case_name]
        if not plan:
            raise CaptureError(f"profile case is outside the selected plan: {case_name}")
    return plan


def capture(campaign_dir: Path, *, sections: tuple[str, ...] = ("memory_paths", "umma_throughput"),
            ncu_binary: str = "ncu", dry_run: bool = False,
            case_name: str | None = None) -> dict:
    """Profiles the planned cases and writes ``<campaign>/ncu/``.  Returns the
    index document that was written."""
    campaign_dir = campaign_dir.resolve()
    if not campaign_dir.is_dir():
        raise CaptureError(f"campaign directory does not exist: {campaign_dir}")
    output_dir = campaign_dir / DIRECTORY
    if output_dir.exists():
        raise CaptureError(f"profile directory already exists: {output_dir}")
    output_dir.mkdir()

    started = utc_now()
    cases = []
    for entry in plan_for(sections, case_name=case_name):
        print(f"ncu: {entry['case']}", file=sys.stderr, flush=True)
        record, exported = capture_case(entry, ncu_binary=ncu_binary, dry_run=dry_run)
        if record["status"] == "captured":
            (output_dir / record["csv"]).write_text(exported, encoding="utf-8")
        else:
            print(f"ncu: {entry['case']}: unavailable ({record['reason']})",
                  file=sys.stderr, flush=True)
        cases.append(record)

    captured = sum(1 for record in cases if record["status"] == "captured")
    if not cases:
        state = "UNAVAILABLE"
    elif captured == len(cases):
        state = "COMPLETE"
    elif captured:
        state = "PARTIAL"
    else:
        state = "UNAVAILABLE"
    index = {
        "schema_version": INDEX_SCHEMA,
        "state": state,
        "started_utc": started,
        "finished_utc": utc_now(),
        "case_count": len(cases),
        "captured_count": captured,
        "metrics": {"memory_paths": list(MEMORY_METRICS), "umma_throughput": list(UMMA_METRICS)},
        "cases": cases,
    }
    (output_dir / INDEX_NAME).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ncu: {captured}/{len(cases)} profiles captured ({state})", file=sys.stderr)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the planned memory and UMMA kernels of one campaign with "
                    "Nsight Compute and store the exported CSV inside the campaign."
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--section", action="append", dest="sections",
                        choices=("memory_paths", "umma_throughput"), default=[])
    parser.add_argument(
        "--case",
        choices=tuple(entry["case"] for entry in (*MEMORY_PLAN, *UMMA_PLAN)),
        help="profile one planned case as a pre-pilot NCU gate",
    )
    parser.add_argument("--ncu-binary", default=os.environ.get("NCU_BINARY", "ncu"))
    parser.add_argument("--dry-run", action="store_true",
                        help="write the index without launching the profiler")
    parser.add_argument(
        "--require-complete", action="store_true",
        help="exit non-zero unless every selected case was captured and parsed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.case and args.sections:
        raise CaptureError("--case cannot be combined with --section")
    sections = tuple(args.sections) or ("memory_paths", "umma_throughput")
    index = capture(
        args.campaign_dir, sections=sections, ncu_binary=args.ncu_binary,
        dry_run=args.dry_run, case_name=args.case)
    if args.require_complete and index["state"] != "COMPLETE":
        raise CaptureError(
            f"required complete NCU capture, observed {index['state']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CaptureError as exc:
        print(f"ncu: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
