#!/usr/bin/env python3
"""Run one complete GB300 campaign and freeze its raw evidence.

The script deliberately owns only four responsibilities: require a clean Git
revision, run the three public benchmarks in a fixed order, validate their raw
CSV contracts, and write a small hash-pinned manifest.  Cross-campaign
statistics belong to ``analysis/analyze.py``.
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
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MEMORY_METHODS = ("ldgsts", "tma")
MEMORY_STAGES = (2, 4, 8)
MEMORY_BIF_BYTES = (16 * 1024, 32 * 1024, 64 * 1024)
UMMA_METHODS = ("umma_1sm", "umma_2sm")
UMMA_N = (64, 128, 256)
UMMA_DEPTHS = (4, 16, 64, 256)
GEMM_SHAPES = tuple(range(1, 6))
GEMM_CANDIDATES = tuple(range(1, 5))

RAW_FILES = (
    "raw/memory_paths.csv",
    "raw/umma_throughput.csv",
    "raw/gemm_comparison.csv",
)

SOURCE_FILES = (
    "Dockerfile",
    "Makefile",
    "VERSIONS.env",
    "PHASE3_VERSIONS.env",
    "memory_paths/Makefile",
    "memory_paths/ldgsts.cu",
    "memory_paths/tma.cu",
    "memory_paths/benchmark.py",
    "umma_throughput/Makefile",
    "umma_throughput/umma_1sm.cu",
    "umma_throughput/umma_2sm.cu",
    "umma_throughput/benchmark.py",
    "gemm_comparison/Makefile",
    "gemm_comparison/gemm_comparison.py",
    "gemm_comparison/cublaslt_bridge.cu",
    "analysis/analyze.py",
    "scripts/run_gpu.sh",
    "scripts/run_campaign.py",
    "scripts/ncu_capture.py",
)

NCU_DIRECTORY = "ncu"


class CampaignError(RuntimeError):
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
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=capture,
        check=False,
    )
    if completed.returncode:
        if capture:
            if completed.stdout:
                sys.stderr.write(completed.stdout)
            if completed.stderr:
                sys.stderr.write(completed.stderr)
        raise CampaignError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed


def git_identity() -> tuple[str, str]:
    top = run(["git", "rev-parse", "--show-toplevel"], capture=True).stdout.strip()
    if Path(top).resolve() != ROOT:
        raise CampaignError(f"Git root is {top}, expected {ROOT}")
    commit = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CampaignError("HEAD is not a canonical Git commit")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=normal"], capture=True
    ).stdout.strip()
    if status:
        raise CampaignError("the repository must be clean before collecting evidence")
    return commit, status


def selected_gpu(index: str) -> dict[str, str]:
    completed = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader",
        ],
        capture=True,
    )
    matches = []
    for line in completed.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 4 and cells[0] == index:
            matches.append(cells)
    if len(matches) != 1:
        raise CampaignError(f"GPU index {index} did not resolve uniquely")
    _, uuid, name, driver = matches[0]
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise CampaignError(f"unexpected GPU UUID {uuid!r}")
    return {"index": index, "uuid": uuid, "name": name, "driver_version": driver}


def image_identity(tag: str) -> dict[str, str]:
    image_id = run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture=True
    ).stdout.strip()
    if not image_id.startswith("sha256:") or not SHA256_RE.fullmatch(image_id[7:]):
        raise CampaignError(f"unexpected Docker image identity {image_id!r}")
    return {"tag": tag, "id": image_id}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields or len(fields) != len(set(fields)):
            raise CampaignError(f"{path}: absent or duplicate CSV header")
        rows = list(reader)
    if not rows:
        raise CampaignError(f"{path}: no data rows")
    if any(None in row or None in row.values() for row in rows):
        raise CampaignError(f"{path}: malformed CSV row")
    return fields, rows


def require_fields(path: Path, fields: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise CampaignError(f"{path}: missing columns {missing}")


def integer(row: dict[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as exc:
        raise CampaignError(f"{path}: invalid integer {field}={row.get(field)!r}") from exc


def finite_positive(row: dict[str, str], field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise CampaignError(f"{path}: invalid number {field}={row.get(field)!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise CampaignError(f"{path}: {field} must be finite and positive")
    return value


def validate_provenance(
    rows: list[dict[str, str]], path: Path, *, commit: str, gpu_uuid: str
) -> None:
    for row_number, row in enumerate(rows, 2):
        if row.get("git_commit") != commit or row.get("git_dirty") != "false":
            raise CampaignError(f"{path}:{row_number}: Git provenance mismatch")
        if row.get("gpu_uuid") != gpu_uuid:
            raise CampaignError(f"{path}:{row_number}: GPU UUID mismatch")


def validate_memory(path: Path, *, commit: str, gpu_uuid: str) -> int:
    fields, rows = read_csv(path)
    require_fields(
        path,
        fields,
        {
            "run_kind", "method", "sample_index", "stages",
            "bytes_in_flight_per_sm", "kernel_time_ms", "effective_gbps",
            "correctness", "mismatches", "gpu_uuid", "git_commit", "git_dirty",
        },
    )
    expected = {
        (method, stages, bif)
        for stages in MEMORY_STAGES
        for bif in MEMORY_BIF_BYTES
        for method in MEMORY_METHODS
    }
    counts: Counter[tuple[str, int, int]] = Counter()
    samples: dict[tuple[str, int, int], set[int]] = {key: set() for key in expected}
    for row in rows:
        if row["run_kind"] != "benchmark" or row["correctness"] != "OK":
            raise CampaignError(f"{path}: non-benchmark or incorrect memory row")
        if integer(row, "mismatches", path) != 0:
            raise CampaignError(f"{path}: non-zero memory mismatches")
        key = (
            row["method"], integer(row, "stages", path),
            integer(row, "bytes_in_flight_per_sm", path),
        )
        if key not in expected:
            raise CampaignError(f"{path}: unexpected memory configuration {key}")
        sample = integer(row, "sample_index", path)
        if sample in samples[key]:
            raise CampaignError(f"{path}: duplicate memory sample {key + (sample,)}")
        samples[key].add(sample)
        counts[key] += 1
        finite_positive(row, "kernel_time_ms", path)
        finite_positive(row, "effective_gbps", path)
    if set(counts) != expected or any(counts[key] != 30 for key in expected):
        raise CampaignError(f"{path}: expected 30 samples for each of 18 configurations")
    if any(samples[key] != set(range(30)) for key in expected):
        raise CampaignError(f"{path}: memory sample indexes are not exactly 0..29")
    validate_provenance(rows, path, commit=commit, gpu_uuid=gpu_uuid)
    return len(rows)


def validate_umma(path: Path, *, commit: str, gpu_uuid: str) -> int:
    fields, rows = read_csv(path)
    require_fields(
        path,
        fields,
        {
            "run_kind", "publishable", "method", "sample_index", "cta_group",
            "n", "depth", "elapsed_cycles", "flops_per_cycle", "correctness",
            "mismatches", "gpu_uuid", "git_commit", "git_dirty",
        },
    )
    expected = {
        (method, n, depth)
        for n in UMMA_N
        for depth in UMMA_DEPTHS
        for method in UMMA_METHODS
    }
    counts: Counter[tuple[str, int, int]] = Counter()
    samples: dict[tuple[str, int, int], set[int]] = {key: set() for key in expected}
    for row in rows:
        if (
            row["run_kind"] != "benchmark"
            or row["publishable"] != "false"
            or row["correctness"] != "OK"
        ):
            raise CampaignError(f"{path}: invalid UMMA row state")
        if integer(row, "mismatches", path) != 0:
            raise CampaignError(f"{path}: non-zero UMMA mismatches")
        key = (row["method"], integer(row, "n", path), integer(row, "depth", path))
        if key not in expected:
            raise CampaignError(f"{path}: unexpected UMMA configuration {key}")
        expected_group = 1 if row["method"] == "umma_1sm" else 2
        if integer(row, "cta_group", path) != expected_group:
            raise CampaignError(f"{path}: method/cta_group mismatch")
        sample = integer(row, "sample_index", path)
        if sample in samples[key]:
            raise CampaignError(f"{path}: duplicate UMMA sample {key + (sample,)}")
        samples[key].add(sample)
        counts[key] += 1
        finite_positive(row, "elapsed_cycles", path)
        finite_positive(row, "flops_per_cycle", path)
    if set(counts) != expected or any(counts[key] != 30 for key in expected):
        raise CampaignError(f"{path}: expected 30 samples for each of 24 configurations")
    if any(samples[key] != set(range(30)) for key in expected):
        raise CampaignError(f"{path}: UMMA sample indexes are not exactly 0..29")
    validate_provenance(rows, path, commit=commit, gpu_uuid=gpu_uuid)
    return len(rows)


def validate_gemm(path: Path, *, commit: str, gpu_uuid: str) -> int:
    fields, rows = read_csv(path)
    require_fields(
        path,
        fields,
        {
            "schema_version", "run_kind", "shape_index", "shape_id",
            "candidate_index", "method", "variant", "correctness",
            "kernel_time_ms", "tflops", "throughput_ratio_vs_cublaslt",
            "gap_to_cublaslt_pct", "best_cutedsl_variant", "gpu_uuid",
            "git_commit", "git_dirty", "publishable",
        },
    )
    expected = {(shape, candidate) for shape in GEMM_SHAPES for candidate in GEMM_CANDIDATES}
    observed = set()
    for row in rows:
        if (
            row["schema_version"] != "p35.v1"
            or row["run_kind"] != "smoke"
            or row["publishable"] != "false"
            or row["correctness"] != "PASS"
        ):
            raise CampaignError(f"{path}: invalid GEMM row state")
        key = (integer(row, "shape_index", path), integer(row, "candidate_index", path))
        if key not in expected or key in observed:
            raise CampaignError(f"{path}: unexpected or duplicate GEMM key {key}")
        observed.add(key)
        finite_positive(row, "kernel_time_ms", path)
        finite_positive(row, "tflops", path)
        finite_positive(row, "throughput_ratio_vs_cublaslt", path)
        try:
            gap = float(row["gap_to_cublaslt_pct"])
        except ValueError as exc:
            raise CampaignError(f"{path}: invalid GEMM gap") from exc
        if not math.isfinite(gap):
            raise CampaignError(f"{path}: non-finite GEMM gap")
    if observed != expected:
        raise CampaignError(f"{path}: expected exactly five shapes x four candidates")
    validate_provenance(rows, path, commit=commit, gpu_uuid=gpu_uuid)
    return len(rows)


def ncu_stage(partial_dir: Path) -> dict:
    """Runs the optional Nsight Compute stage.  The stage is advisory: a
    profiler that is absent, refused counter access or failed on one case is
    recorded and the campaign continues, because the timing evidence the
    campaign exists to produce does not depend on it."""
    block = {"requested": True, "state": "UNAVAILABLE", "case_count": 0,
             "captured_count": 0, "artifact_sha256": {}}
    try:
        run([sys.executable, "scripts/ncu_capture.py", "--campaign-dir", str(partial_dir)])
    except CampaignError as exc:
        block["reason"] = str(exc)
    directory = partial_dir / NCU_DIRECTORY
    index_path = directory / "index.json"
    if not index_path.is_file():
        block.setdefault("reason", "the profile stage produced no index")
        return block
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        block["reason"] = f"unreadable profile index: {exc}"
        return block
    block["state"] = index.get("state", "UNAVAILABLE")
    block["case_count"] = index.get("case_count", 0)
    block["captured_count"] = index.get("captured_count", 0)
    block["artifact_sha256"] = {
        str(path.relative_to(partial_dir)): sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    }
    return block


def source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise CampaignError(f"required source file is absent: {relative}")
        result[relative] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run memory, UMMA and GEMM once and freeze one campaign directory."
    )
    parser.add_argument("--kind", choices=("pilot", "final"), required=True)
    parser.add_argument("--campaign-id", help="UTC identifier YYYYMMDDTHHMMSSZ")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--with-ncu", action="store_true",
        help="also profile the planned kernels with Nsight Compute; never fatal",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_id = args.campaign_id or default_campaign_id()
    if not CAMPAIGN_RE.fullmatch(campaign_id):
        raise CampaignError("campaign ID must match YYYYMMDDTHHMMSSZ")
    gpu_index = os.environ.get("BLACKWELL_GPU_INDEX", "")
    if not re.fullmatch(r"[0-9]+", gpu_index):
        raise CampaignError("BLACKWELL_GPU_INDEX must be set explicitly")

    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root = output_root.resolve()
    try:
        output_root.relative_to(ROOT)
    except ValueError as exc:
        raise CampaignError("output root must be inside the repository") from exc

    final_dir = output_root / campaign_id
    partial_dir = output_root / f"{campaign_id}.partial"
    if final_dir.exists() or partial_dir.exists():
        raise CampaignError(f"campaign path already exists for {campaign_id}")

    commit, _ = git_identity()
    gpu = selected_gpu(gpu_index)
    image_tag = os.environ.get("IMAGE_TAG", "gb300-microbench:latest")
    image = image_identity(image_tag)
    started = utc_now()

    raw_dir = partial_dir / "raw"
    raw_dir.mkdir(parents=True)
    memory_path = raw_dir / "memory_paths.csv"
    umma_path = raw_dir / "umma_throughput.csv"
    gemm_path = raw_dir / "gemm_comparison.csv"

    run(
        [
            sys.executable, "memory_paths/benchmark.py", "--run-kind", "benchmark",
            "--output", str(memory_path),
        ]
    )
    run(
        [
            sys.executable, "umma_throughput/benchmark.py", "--run-kind", "benchmark",
            "--output", str(umma_path),
        ]
    )
    gemm_relative = os.path.relpath(gemm_path, ROOT / "gemm_comparison")
    run(
        [
            str(ROOT / "scripts" / "run_gpu.sh"), "make", "-C", "gemm_comparison",
            "run", f"ARCH={os.environ.get('CUDA_ARCH', 'sm_103a')}",
            f"OUTPUT={gemm_relative}", "WARMUP=2", "ITERATIONS=10",
        ]
    )

    row_counts = {
        "memory_paths": validate_memory(memory_path, commit=commit, gpu_uuid=gpu["uuid"]),
        "umma_throughput": validate_umma(umma_path, commit=commit, gpu_uuid=gpu["uuid"]),
        "gemm_comparison": validate_gemm(gemm_path, commit=commit, gpu_uuid=gpu["uuid"]),
    }
    ncu = (ncu_stage(partial_dir) if args.with_ncu
           else {"requested": False, "state": "NOT_REQUESTED", "case_count": 0,
                 "captured_count": 0, "artifact_sha256": {}})
    artifacts = {relative: sha256_file(partial_dir / relative) for relative in RAW_FILES}
    finished = utc_now()
    manifest = {
        "schema_version": "gb300.campaign.v1",
        "campaign_id": campaign_id,
        "campaign_kind": args.kind,
        "state": "COMPLETE",
        "started_utc": iso_utc(started),
        "finished_utc": iso_utc(finished),
        "git_commit": commit,
        "git_dirty": False,
        "gpu": gpu,
        "container_image": image,
        "stage_order": ["memory_paths", "umma_throughput", "gemm_comparison"]
                       + (["ncu_capture"] if args.with_ncu else []),
        "parameters": {
            "memory_paths": {
                "working_set_mib": 512, "passes": 32, "warmup_ms": 2000,
                "repetitions": 30, "configuration_count": 18,
            },
            "umma_throughput": {
                "iterations": 1000, "warmup_iterations": 10,
                "repetitions": 30, "configuration_count": 24,
            },
            "gemm_comparison": {
                "warmup_iterations": 2, "iterations": 10,
                "shape_count": 5, "candidate_count": 4, "cache_mode": "hot",
            },
        },
        "row_counts": row_counts,
        "artifact_sha256": artifacts,
        "ncu": ncu,
        "source_sha256": source_hashes(),
    }
    write_json(partial_dir / "manifest.json", manifest)
    (partial_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(artifacts.items())),
        encoding="utf-8",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    partial_dir.rename(final_dir)
    print(f"campaign: COMPLETE {final_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as exc:
        print(f"campaign: ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
