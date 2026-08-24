#!/usr/bin/env python3
"""Freeze one supplementary UMMA device-scaling campaign.

The supplementary experiment is deliberately separate from the closed
three-experiment campaign contract in ``scripts/run_campaign.py``: it lives
under its own output root, writes its own raw CSV schema, and never touches
``runs/<UTC-ID>/`` or ``results/new``.  This module also owns the raw-CSV
contract, which ``analysis/analyze_umma_device_scaling.py`` imports.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SCHEMA = "gb300.umma-scaling-campaign.v1"
ROW_SCHEMA = "umma-device-scaling.v1"

METHODS = ("umma_1sm", "umma_2sm")
SCALES = ("isolated", "device_scale")
# Canonical order; the binary alternates its execution order between
# repetitions, which is why every row also carries execution_order.
CONFIGURATIONS = (
    ("umma_1sm", "isolated"),
    ("umma_2sm", "isolated"),
    ("umma_1sm", "device_scale"),
    ("umma_2sm", "device_scale"),
)
DEVICE_SCALE_CONFIGURATIONS = (("umma_1sm", "device_scale"), ("umma_2sm", "device_scale"))

FROZEN_N = 256
FROZEN_K = 16
FROZEN_DEPTH = 256
FROZEN_THREADS_PER_CTA = 128
BENCHMARK_ITERATIONS = 1000
BENCHMARK_WARMUP = 10
BENCHMARK_REPETITIONS = 30

RAW_FILE = "raw/umma_device_scaling.csv"
RAW_FILES = (RAW_FILE,)

SOURCE_FILES = (
    "Dockerfile",
    "Makefile",
    "VERSIONS.env",
    "umma_throughput/Makefile",
    "umma_throughput/umma_device_scaling.cu",
    "umma_throughput/device_scaling.py",
    "scripts/run_gpu.sh",
    "scripts/run_umma_scaling_campaign.py",
    "analysis/analyze_umma_device_scaling.py",
)

# Per-configuration columns that must be constant inside one campaign.
CONSTANT_FIELDS = (
    "cta_group", "m", "n", "k", "depth", "iterations", "warmup_iterations", "repetitions",
    "work_unit_count", "umma_per_work_unit_per_iteration", "total_umma_count", "flops_per_umma",
    "total_flops", "threads_per_cta", "grid_blocks", "cluster_size", "cluster_count",
    "hardware_sm_count", "planned_active_sm_count", "occupancy_blocks_per_sm",
    "max_active_clusters", "shared_memory_reservation_bytes", "unused_sm_count",
    "coverage_status", "residency_evidence", "operand_path", "input_type", "accumulator_type",
)

REQUIRED_FIELDS = frozenset(CONSTANT_FIELDS) | {
    "schema_version", "timestamp_utc", "campaign_kind", "run_kind", "publishable",
    "sample_index", "execution_order", "method", "scale", "kernel_time_ms", "total_tflops",
    "tflops_per_planned_active_sm", "tflops_per_evidenced_active_sm",
    "observed_unique_sm_count", "correctness", "mismatches", "max_abs_error",
    "gpu_name", "gpu_uuid", "compute_capability", "cuda_driver_version",
    "cuda_runtime_version", "git_commit", "git_dirty",
    "diagnostic_clock64_cycles_min", "diagnostic_clock64_cycles_max",
}

FROZEN_LABELS = {"operand_path": "smem_smem", "input_type": "bf16", "accumulator_type": "fp32"}
FROZEN_PARAMETERS = {
    "n": FROZEN_N,
    "k": FROZEN_K,
    "depth": FROZEN_DEPTH,
    "threads_per_cta": FROZEN_THREADS_PER_CTA,
    "iterations": BENCHMARK_ITERATIONS,
    "warmup_iterations": BENCHMARK_WARMUP,
    "repetitions": BENCHMARK_REPETITIONS,
    "configuration_count": len(CONFIGURATIONS),
    **FROZEN_LABELS,
    "timing_source": "cuda_event_whole_kernel",
}

COVERAGE_ACCEPTED = ("full_device_coverage", "maximum_resident_coverage")


class ScalingCampaignError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso_utc(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def default_campaign_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    completed = subprocess.run(
        command, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=capture, check=False
    )
    if completed.returncode:
        if capture:
            if completed.stdout:
                sys.stderr.write(completed.stdout)
            if completed.stderr:
                sys.stderr.write(completed.stderr)
        raise ScalingCampaignError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed


def git_identity() -> str:
    top = run(["git", "rev-parse", "--show-toplevel"], capture=True).stdout.strip()
    if Path(top).resolve() != ROOT:
        raise ScalingCampaignError(f"Git root is {top}, expected {ROOT}")
    commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ScalingCampaignError("HEAD is not a canonical Git commit")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=normal"], capture=True
    ).stdout.strip()
    if status:
        raise ScalingCampaignError("the repository must be clean before collecting evidence")
    return commit


def selected_gpu(index: str) -> dict[str, str]:
    completed = run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        capture=True,
    )
    matches = []
    for line in completed.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 4 and cells[0] == index:
            matches.append(cells)
    if len(matches) != 1:
        raise ScalingCampaignError(f"GPU index {index} did not resolve uniquely")
    _, uuid, name, driver = matches[0]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise ScalingCampaignError(f"unexpected GPU UUID {uuid!r}")
    return {"index": index, "uuid": uuid, "name": name, "driver_version": driver}


def image_identity(tag: str) -> dict[str, str]:
    image_id = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture=True
    ).stdout.strip()
    if not image_id.startswith("sha256:") or not SHA256_RE.fullmatch(image_id[7:]):
        raise ScalingCampaignError(f"unexpected Docker image identity {image_id!r}")
    return {"tag": tag, "id": image_id}


# ---------------------------------------------------------------------------
# Raw CSV contract.
# ---------------------------------------------------------------------------
def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if not fields or len(fields) != len(set(fields)):
                raise ScalingCampaignError(f"{path}: absent or duplicate CSV header")
            rows = list(reader)
    except OSError as exc:
        raise ScalingCampaignError(f"{path}: cannot read raw CSV: {exc}") from exc
    if not rows:
        raise ScalingCampaignError(f"{path}: no data rows")
    if any(None in row or None in row.values() for row in rows):
        raise ScalingCampaignError(f"{path}: malformed CSV row")
    return fields, rows


def integer(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise ScalingCampaignError(
            f"{path}: invalid integer {field}={row.get(field)!r}"
        ) from exc


def finite_positive(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ScalingCampaignError(f"{path}: invalid number {field}={row.get(field)!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ScalingCampaignError(f"{path}: {field} must be finite and positive")
    return value


def validate_scaling_csv(
    path: Path, *, commit: str, gpu_uuid: str, campaign_kind: str, repetitions: int
) -> dict:
    """Validates the exact four-configuration matrix and returns its facts."""
    if repetitions != BENCHMARK_REPETITIONS:
        raise ScalingCampaignError(f"{path}: repetitions must be frozen at {BENCHMARK_REPETITIONS}")
    fields, rows = read_csv(path)
    missing = sorted(REQUIRED_FIELDS - set(fields))
    if missing:
        raise ScalingCampaignError(f"{path}: missing columns {missing}")

    expected = set(CONFIGURATIONS)
    samples: dict[tuple[str, str], set[int]] = {key: set() for key in expected}
    constants: dict[tuple[str, str], dict[str, str]] = {}
    for number, row in enumerate(rows, 2):
        where = f"{path}:{number}"
        if row["schema_version"] != ROW_SCHEMA:
            raise ScalingCampaignError(f"{where}: unsupported row schema {row['schema_version']!r}")
        if row["run_kind"] != "benchmark" or row["publishable"] != "false":
            raise ScalingCampaignError(f"{where}: invalid row state")
        if row["campaign_kind"] != campaign_kind:
            raise ScalingCampaignError(
                f"{where}: campaign_kind {row['campaign_kind']!r} != {campaign_kind!r}"
            )
        if row["correctness"] != "OK":
            raise ScalingCampaignError(f"{where}: correctness is not OK")
        if integer(row, "mismatches", path) != 0:
            raise ScalingCampaignError(f"{where}: non-zero mismatches")
        if float(row["max_abs_error"]) != 0.0:
            raise ScalingCampaignError(f"{where}: non-zero max_abs_error")
        if row["git_commit"] != commit or row["git_dirty"] != "false":
            raise ScalingCampaignError(f"{where}: Git provenance mismatch")
        if row["gpu_uuid"] != gpu_uuid:
            raise ScalingCampaignError(f"{where}: GPU UUID mismatch")

        key = (row["method"], row["scale"])
        if key not in expected:
            raise ScalingCampaignError(f"{where}: unexpected configuration {key}")
        sample = integer(row, "sample_index", path)
        if sample in samples[key]:
            raise ScalingCampaignError(f"{path}: duplicate sample {key + (sample,)}")
        samples[key].add(sample)
        position = CONFIGURATIONS.index(key)
        step = position if sample % 2 == 0 else len(CONFIGURATIONS) - position - 1
        if integer(row, "execution_order", path) != sample * len(CONFIGURATIONS) + step:
            raise ScalingCampaignError(f"{where}: execution_order does not alternate by repetition")

        if integer(row, "n", path) != FROZEN_N or integer(row, "k", path) != FROZEN_K:
            raise ScalingCampaignError(f"{where}: N/K are not the frozen {FROZEN_N}/{FROZEN_K}")
        if integer(row, "depth", path) != FROZEN_DEPTH:
            raise ScalingCampaignError(f"{where}: depth is not the frozen {FROZEN_DEPTH}")
        if integer(row, "threads_per_cta", path) != FROZEN_THREADS_PER_CTA:
            raise ScalingCampaignError(f"{where}: threads_per_cta is not {FROZEN_THREADS_PER_CTA}")
        group = 1 if row["method"] == "umma_1sm" else 2
        for field, expected_value in (
            ("m", 128 * group),
            ("iterations", BENCHMARK_ITERATIONS),
            ("warmup_iterations", BENCHMARK_WARMUP),
            ("umma_per_work_unit_per_iteration", FROZEN_DEPTH),
        ):
            if integer(row, field, path) != expected_value:
                raise ScalingCampaignError(f"{where}: {field} must be {expected_value}")
        for label, expected_value in FROZEN_LABELS.items():
            if row[label] != expected_value:
                raise ScalingCampaignError(
                    f"{where}: {label} is {row[label]!r}, expected {expected_value!r}")
        if integer(row, "cta_group", path) != group:
            raise ScalingCampaignError(f"{where}: method/cta_group mismatch")
        if integer(row, "occupancy_blocks_per_sm", path) != 1:
            raise ScalingCampaignError(f"{where}: occupancy is not one resident CTA per SM")
        if integer(row, "repetitions", path) != repetitions:
            raise ScalingCampaignError(f"{where}: repetitions column does not match {repetitions}")

        hardware = integer(row, "hardware_sm_count", path)
        planned = integer(row, "planned_active_sm_count", path)
        observed = integer(row, "observed_unique_sm_count", path)
        work_units = integer(row, "work_unit_count", path)
        if (
            not 1 <= planned <= hardware
            or not 0 <= observed <= planned
            or integer(row, "grid_blocks", path) != planned
            or integer(row, "cluster_size", path) != group
            or planned != work_units * group
            or integer(row, "unused_sm_count", path) != hardware - planned
        ):
            raise ScalingCampaignError(f"{where}: inconsistent launch geometry or SM coverage")
        if group == 1:
            if (
                row["cluster_count"] != "not_applicable"
                or row["max_active_clusters"] != "not_applicable"
            ):
                raise ScalingCampaignError(f"{where}: 1-SM launch cannot declare clusters")
        else:
            max_clusters = integer(row, "max_active_clusters", path)
            if integer(row, "cluster_count", path) != work_units or max_clusters < work_units:
                raise ScalingCampaignError(f"{where}: inconsistent 2-SM cluster geometry")
        if row["scale"] == "isolated" and work_units != 1:
            raise ScalingCampaignError(f"{where}: isolated launch must contain one work unit")
        if row["scale"] == "device_scale" and work_units != (
            hardware if group == 1 else min(hardware // 2, max_clusters)
        ):
            raise ScalingCampaignError(f"{where}: device launch does not use maximum resident coverage")

        flops_per_umma = 2 * (128 * group) * FROZEN_N * FROZEN_K
        total_umma = FROZEN_DEPTH * BENCHMARK_ITERATIONS * work_units
        total_flops = flops_per_umma * total_umma
        for field, expected_value in (
            ("flops_per_umma", flops_per_umma),
            ("total_umma_count", total_umma),
            ("total_flops", total_flops),
        ):
            if integer(row, field, path) != expected_value:
                raise ScalingCampaignError(f"{where}: {field} does not match FLOP accounting")
        kernel_time = finite_positive(row, "kernel_time_ms", path)
        total_tflops = finite_positive(row, "total_tflops", path)
        for field, expected_value in (
            ("total_tflops", total_flops / kernel_time / 1e9),
            ("tflops_per_planned_active_sm", total_tflops / planned),
        ):
            if not math.isclose(
                finite_positive(row, field, path), expected_value, rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ScalingCampaignError(f"{where}: {field} does not match FLOP/time accounting")

        coverage = row["coverage_status"]
        if row["scale"] == "device_scale":
            if coverage not in COVERAGE_ACCEPTED:
                raise ScalingCampaignError(
                    f"{where}: device-scale coverage_status is {coverage!r}; "
                    "an unproven device-scale launch cannot be published"
                )
            if row["residency_evidence"] != "all_blocks_simultaneously_resident":
                raise ScalingCampaignError(f"{where}: device-scale residency was not established")
            if observed != planned:
                raise ScalingCampaignError(f"{where}: observed SM count != planned SM count")
            expected_coverage = (
                "full_device_coverage" if planned == hardware else "maximum_resident_coverage"
            )
            if coverage != expected_coverage:
                raise ScalingCampaignError(f"{where}: coverage_status does not match active SM count")
        elif coverage != "isolated_unit":
            raise ScalingCampaignError(f"{where}: isolated coverage_status is {coverage!r}")

        evidenced = row["tflops_per_evidenced_active_sm"]
        if evidenced == "not_applicable":
            if row["scale"] == "device_scale":
                raise ScalingCampaignError(f"{where}: evidenced active-SM throughput is missing")
        elif observed == 0 or not math.isclose(
            finite_positive(row, "tflops_per_evidenced_active_sm", path), total_tflops / observed,
            rel_tol=1e-5, abs_tol=1e-6,
        ):
            raise ScalingCampaignError(f"{where}: evidenced active-SM throughput does not match")

        snapshot = {field: row[field] for field in CONSTANT_FIELDS}
        if key not in constants:
            constants[key] = snapshot
        elif constants[key] != snapshot:
            differing = sorted(f for f in CONSTANT_FIELDS if constants[key][f] != snapshot[f])
            raise ScalingCampaignError(f"{where}: {key} changed {differing} between samples")

    if set(constants) != expected:
        missing_keys = sorted(expected - set(constants))
        raise ScalingCampaignError(f"{path}: missing configuration(s) {missing_keys}")
    for key in sorted(expected):
        if samples[key] != set(range(repetitions)):
            raise ScalingCampaignError(
                f"{path}: {key} sample indexes are not exactly 0..{repetitions - 1}"
            )
    hardware = {constants[key]["hardware_sm_count"] for key in expected}
    if len(hardware) != 1:
        raise ScalingCampaignError(f"{path}: rows disagree on hardware_sm_count")
    reservation = {constants[key]["shared_memory_reservation_bytes"] for key in expected}
    if len(reservation) != 1:
        raise ScalingCampaignError(f"{path}: rows disagree on shared_memory_reservation_bytes")

    one = constants[("umma_1sm", "device_scale")]
    two = constants[("umma_2sm", "device_scale")]
    planned_one, planned_two = int(one["planned_active_sm_count"]), int(two["planned_active_sm_count"])
    if planned_one == planned_two and planned_one % 2 == 0:
        if int(one["total_flops"]) != int(two["total_flops"]):
            raise ScalingCampaignError(
                f"{path}: equal active-SM coverage ({planned_one} SMs) must give equal total FLOPs, "
                f"found {one['total_flops']} vs {two['total_flops']}"
            )

    return {
        "row_count": len(rows),
        "hardware_sm_count": int(hardware.pop()),
        "shared_memory_reservation_bytes": int(reservation.pop()),
        "equal_active_sm_coverage": planned_one == planned_two,
        "configurations": {
            f"{method}/{scale}": {
                "grid_blocks": int(constants[(method, scale)]["grid_blocks"]),
                "cluster_size": int(constants[(method, scale)]["cluster_size"]),
                "work_unit_count": int(constants[(method, scale)]["work_unit_count"]),
                "planned_active_sm_count": int(constants[(method, scale)]["planned_active_sm_count"]),
                "unused_sm_count": int(constants[(method, scale)]["unused_sm_count"]),
                "total_flops": int(constants[(method, scale)]["total_flops"]),
                "coverage_status": constants[(method, scale)]["coverage_status"],
                "max_active_clusters": constants[(method, scale)]["max_active_clusters"],
            }
            for method, scale in CONFIGURATIONS
        },
    }


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ScalingCampaignError(f"required source file is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the supplementary UMMA device-scaling experiment once and freeze it."
    )
    parser.add_argument("--kind", choices=("pilot", "final"), required=True)
    parser.add_argument("--campaign-id", help="UTC identifier YYYYMMDDTHHMMSSZ")
    parser.add_argument("--output-root", type=Path, default=Path("runs/umma_device_scaling"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_id = args.campaign_id or default_campaign_id()
    if not CAMPAIGN_RE.fullmatch(campaign_id):
        raise ScalingCampaignError("campaign ID must match YYYYMMDDTHHMMSSZ")
    gpu_index = os.environ.get("BLACKWELL_GPU_INDEX", "")
    if not re.fullmatch(r"[0-9]+", gpu_index):
        raise ScalingCampaignError("BLACKWELL_GPU_INDEX must be set explicitly")

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(ROOT)
    except ValueError as exc:
        raise ScalingCampaignError("output root must be inside the repository") from exc

    final_dir = output_root / campaign_id
    partial_dir = output_root / f"{campaign_id}.partial"
    if final_dir.exists() or partial_dir.exists():
        raise ScalingCampaignError(f"campaign path already exists for {campaign_id}")

    commit = git_identity()
    gpu = selected_gpu(gpu_index)
    image = image_identity(os.environ.get("IMAGE_TAG", "gb300-microbench:latest"))
    started = utc_now()

    raw_dir = partial_dir / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = partial_dir / RAW_FILE
    run(
        [
            sys.executable, "umma_throughput/device_scaling.py", "--run-kind", "benchmark",
            "--campaign-kind", args.kind, "--iterations", str(BENCHMARK_ITERATIONS),
            "--warmup-iterations", str(BENCHMARK_WARMUP),
            "--repetitions", str(BENCHMARK_REPETITIONS), "--output", str(raw_path),
        ]
    )

    facts = validate_scaling_csv(
        raw_path, commit=commit, gpu_uuid=gpu["uuid"], campaign_kind=args.kind,
        repetitions=BENCHMARK_REPETITIONS,
    )
    artifacts = {RAW_FILE: sha256_file(raw_path)}
    finished = utc_now()
    manifest = {
        "schema_version": SCHEMA,
        "campaign_id": campaign_id,
        "campaign_kind": args.kind,
        "state": "COMPLETE",
        "started_utc": iso_utc(started),
        "finished_utc": iso_utc(finished),
        "git_commit": commit,
        "git_dirty": False,
        "gpu": gpu,
        "container_image": image,
        "stage_order": ["umma_device_scaling"],
        "parameters": FROZEN_PARAMETERS,
        "device_scaling": facts,
        "row_counts": {"umma_device_scaling": facts["row_count"]},
        "artifact_sha256": artifacts,
        "source_sha256": source_hashes(),
    }
    write_json(partial_dir / "manifest.json", manifest)
    (partial_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(artifacts.items())),
        encoding="utf-8",
    )
    partial_dir.rename(final_dir)
    print(f"umma-scaling-campaign: COMPLETE {final_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScalingCampaignError as exc:
        print(f"umma-scaling-campaign: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
