#!/usr/bin/env python3
"""GPU-free tests for the supplementary UMMA device-scaling analyzer.

Every campaign here is synthetic: the point is to prove that the analyzer
accepts a well-formed three-campaign population and refuses each way one can
be malformed, without needing a GB300.  The raw CSV header is read out of
``umma_device_scaling.cu`` itself, so a schema change in the binary that the
analyzer has not followed makes these tests fail rather than silently pass
against a stale copy of the header.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_umma_scaling_campaign as contract  # noqa: E402

ANALYZER = ROOT / "analysis" / "analyze_umma_device_scaling.py"
SOURCE = ROOT / "umma_throughput" / "umma_device_scaling.cu"

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
GPU_A = {"index": "0", "uuid": "GPU-53ed8fd9-9f5a-9c77-430d-095ad4fc860d",
         "name": "NVIDIA B300 SXM6 AC", "driver_version": "610.43.02"}
GPU_B = {"index": "1", "uuid": "GPU-adecbd14-4768-00f8-a2fa-697ced804a8d",
         "name": "NVIDIA B300 SXM6 AC", "driver_version": "610.43.02"}
IMAGE = {"tag": "gb300-microbench:latest", "id": "sha256:" + "c" * 64}

REPETITIONS = 30
ITERATIONS = 1000
DEPTH = 256
N = 256
K = 16
RESERVATION = 117760
# Deterministic per-configuration throughput, in TFLOP/s, before jitter.
BASE_TFLOPS = {
    ("umma_1sm", "isolated"): 14.5,
    ("umma_2sm", "isolated"): 28.6,
    ("umma_1sm", "device_scale"): 2070.0,
    ("umma_2sm", "device_scale"): 2045.0,
}


def csv_header() -> list[str]:
    """Extracts the raw CSV header from the CUDA source's print_csv_header."""
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("void print_csv_header()")
    body = text[start:text.index("}", start)]
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    header = "".join(pieces).replace("\\n", "")
    fields = header.split(",")
    assert len(fields) == len(set(fields)), "duplicate column in the raw CSV header"
    return fields


HEADER = csv_header()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(sm_count: int) -> dict[tuple[str, str], dict]:
    clusters = sm_count // 2
    return {
        ("umma_1sm", "isolated"): dict(
            cta_group=1, m=128, cluster_size=1, cluster_count="not_applicable", grid_blocks=1,
            work_units=1, planned=1, max_active_clusters="not_applicable",
            coverage="isolated_unit"),
        ("umma_2sm", "isolated"): dict(
            cta_group=2, m=256, cluster_size=2, cluster_count=1, grid_blocks=2, work_units=1,
            planned=2, max_active_clusters=clusters, coverage="isolated_unit"),
        ("umma_1sm", "device_scale"): dict(
            cta_group=1, m=128, cluster_size=1, cluster_count="not_applicable",
            grid_blocks=sm_count, work_units=sm_count, planned=sm_count,
            max_active_clusters="not_applicable",
            coverage="full_device_coverage"),
        ("umma_2sm", "device_scale"): dict(
            cta_group=2, m=256, cluster_size=2, cluster_count=clusters,
            grid_blocks=2 * clusters, work_units=clusters, planned=2 * clusters,
            max_active_clusters=clusters,
            coverage="full_device_coverage" if 2 * clusters == sm_count
            else "maximum_resident_coverage"),
    }


def make_rows(*, commit: str, gpu: dict, campaign_kind: str, sm_count: int,
              offset: float) -> list[dict[str, str]]:
    plan = geometry(sm_count)
    rows = []
    order = 0
    for sample in range(REPETITIONS):
        keys = list(plan)
        if sample % 2:
            keys = list(reversed(keys))
        for method, scale in keys:
            spec = plan[(method, scale)]
            flops_per_umma = 2 * spec["m"] * N * K
            total_umma = DEPTH * ITERATIONS * spec["work_units"]
            total_flops = flops_per_umma * total_umma
            # Deterministic jitter so medians differ slightly per campaign.
            tflops = BASE_TFLOPS[(method, scale)] * (
                1.0 + offset + 0.0004 * ((sample * 7 + order) % 11 - 5))
            kernel_time_ms = total_flops / (tflops * 1e12) * 1e3
            rows.append({
                "schema_version": "umma-device-scaling.v1",
                "timestamp_utc": "2026-08-24T12:00:00Z",
                "campaign_kind": campaign_kind,
                "run_kind": "benchmark",
                "publishable": "false",
                "sample_index": str(sample),
                "execution_order": str(order),
                "method": method,
                "scale": scale,
                "cta_group": str(spec["cta_group"]),
                "m": str(spec["m"]),
                "n": str(N),
                "k": str(K),
                "depth": str(DEPTH),
                "iterations": str(ITERATIONS),
                "warmup_iterations": "10",
                "repetitions": str(REPETITIONS),
                "work_unit_count": str(spec["work_units"]),
                "umma_per_work_unit_per_iteration": str(DEPTH),
                "total_umma_count": str(total_umma),
                "flops_per_umma": str(flops_per_umma),
                "total_flops": str(total_flops),
                "kernel_time_ms": f"{kernel_time_ms:.6f}",
                "total_tflops": f"{tflops:.6f}",
                "tflops_per_planned_active_sm": f"{tflops / spec['planned']:.6f}",
                "tflops_per_evidenced_active_sm": f"{tflops / spec['planned']:.6f}",
                "threads_per_cta": "128",
                "grid_blocks": str(spec["grid_blocks"]),
                "cluster_size": str(spec["cluster_size"]),
                "cluster_count": str(spec["cluster_count"]),
                "hardware_sm_count": str(sm_count),
                "planned_active_sm_count": str(spec["planned"]),
                "observed_unique_sm_count": str(spec["planned"]),
                "occupancy_blocks_per_sm": "1",
                "max_active_clusters": str(spec["max_active_clusters"]),
                "shared_memory_reservation_bytes": str(RESERVATION),
                "unused_sm_count": str(sm_count - spec["planned"]),
                "coverage_status": spec["coverage"],
                "residency_evidence": "all_blocks_simultaneously_resident",
                "diagnostic_clock64_cycles_min": "31000000",
                "diagnostic_clock64_cycles_max": "31900000",
                "operand_path": "smem_smem",
                "input_type": "bf16",
                "accumulator_type": "fp32",
                "correctness": "OK",
                "mismatches": "0",
                "max_abs_error": "0.000000",
                "gpu_name": "NVIDIA B300 SXM6 AC",
                "gpu_uuid": gpu["uuid"],
                "compute_capability": "10.3",
                "cuda_driver_version": "13010",
                "cuda_runtime_version": "13010",
                "git_commit": commit,
                "git_dirty": "false",
            })
            order += 1
    return rows


def freeze(directory: Path, rows: list[dict[str, str]], *, campaign_id: str, commit: str,
           gpu: dict, campaign_kind: str = "final", state: str = "COMPLETE",
           schema: str | None = None, rehash: bool = True) -> None:
    raw = directory / "raw" / "umma_device_scaling.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sm_count = int(rows[0]["hardware_sm_count"])
    plan = geometry(sm_count)
    manifest = {
        "schema_version": schema or "gb300.umma-scaling-campaign.v1",
        "campaign_id": campaign_id,
        "campaign_kind": campaign_kind,
        "state": state,
        "started_utc": "2026-08-24T12:00:00Z",
        "finished_utc": "2026-08-24T12:20:00Z",
        "git_commit": commit,
        "git_dirty": False,
        "gpu": gpu,
        "container_image": IMAGE,
        "stage_order": ["umma_device_scaling"],
        "parameters": {
            "n": N, "k": K, "depth": DEPTH, "threads_per_cta": 128, "iterations": ITERATIONS,
            "warmup_iterations": 10, "repetitions": REPETITIONS, "configuration_count": 4,
            "input_type": "bf16", "accumulator_type": "fp32", "operand_path": "smem_smem",
            "timing_source": "cuda_event_whole_kernel",
        },
        "device_scaling": {
            "row_count": len(rows),
            "hardware_sm_count": sm_count,
            "shared_memory_reservation_bytes": RESERVATION,
            "equal_active_sm_coverage":
                plan[("umma_1sm", "device_scale")]["planned"]
                == plan[("umma_2sm", "device_scale")]["planned"],
        },
        "row_counts": {"umma_device_scaling": len(rows)},
        "artifact_sha256": {"raw/umma_device_scaling.csv":
                            sha256_file(raw) if rehash else "0" * 64},
        "source_sha256": {relative: "d" * 64 for relative in contract.SOURCE_FILES},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "SHA256SUMS").write_text(
        f"{sha256_file(raw)}  raw/umma_device_scaling.csv\n", encoding="utf-8")


def build_population(root: Path, *, sm_count: int = 148) -> list[Path]:
    paths = []
    for index, campaign_id in enumerate(
        ("20260824T120000Z", "20260824T130000Z", "20260824T140000Z")
    ):
        directory = root / campaign_id
        rows = make_rows(commit=COMMIT_A, gpu=GPU_A, campaign_kind="final", sm_count=sm_count,
                         offset=0.001 * index)
        freeze(directory, rows, campaign_id=campaign_id, commit=COMMIT_A, gpu=GPU_A)
        paths.append(directory)
    return paths


def run_analyzer(campaigns: list[Path], output: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ANALYZER)]
    for path in campaigns:
        command += ["--campaign", str(path)]
    command += ["--output", str(output)]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------
FAILURES: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(': ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append((name, detail))


def expect_rejected(name: str, root: Path, campaigns: list[Path], fragment: str) -> None:
    completed = run_analyzer(campaigns, root / "out")
    ok = completed.returncode == 2 and fragment in completed.stderr
    check(name, ok, f"rc={completed.returncode} stderr={completed.stderr.strip()[-300:]!r}")


def case_valid(work: Path) -> None:
    root = work / "valid"
    campaigns = build_population(root)
    output = root / "analysis"
    completed = run_analyzer(campaigns, output)
    check("valid population is accepted", completed.returncode == 0,
          f"rc={completed.returncode} stderr={completed.stderr.strip()[-500:]!r}")
    if completed.returncode != 0:
        return
    for name in ("umma_device_scaling.csv", "summary.json", "manifest.json", "SHA256SUMS",
                 "figures/umma_device_scaling.svg"):
        check(f"valid population writes {name}", (output / name).is_file())

    summary = json.loads((output / "summary.json").read_text())
    check("summary records three campaigns", summary["campaign_count"] == 3)
    check("summary records 148 SMs", summary["hardware_sm_count"] == 148)
    check("summary reports equal active-SM coverage", summary["equal_active_sm_coverage"] is True)
    check("summary reports no warnings", summary["warnings"] == [], str(summary["warnings"]))

    per_campaign = summary["per_campaign_derived"]
    check("one derived block per campaign", len(per_campaign) == 3)
    first = per_campaign[0]
    expected_ideal = first["isolated_1sm_tflops"] * first["active_1sm_blocks"]
    check("1-SM ideal projection uses the measured isolated baseline",
          abs(first["ideal_linear_1sm_tflops"] - expected_ideal) < 1e-6)
    expected_efficiency = 100.0 * first["device_1sm_tflops"] / expected_ideal
    check("1-SM scaling efficiency matches its definition",
          abs(first["scaling_efficiency_1sm_percent"] - expected_efficiency) < 1e-9)
    check("2-SM efficiency divides by the cluster count, not the block count",
          abs(first["ideal_linear_2sm_tflops"]
              - first["isolated_2sm_tflops"] * first["active_2sm_clusters"]) < 1e-6)
    check("active 1-SM blocks and 2-SM clusters differ by exactly two",
          first["active_1sm_blocks"] == 2 * first["active_2sm_clusters"])
    check("gap and efficiency are complementary",
          abs(first["gap_from_ideal_1sm_percent"]
              + first["scaling_efficiency_1sm_percent"] - 100.0) < 1e-9)
    check("device totals ratio is exposed",
          abs(first["device_total_ratio_2sm_over_1sm"]
              - first["device_2sm_tflops"] / first["device_1sm_tflops"]) < 1e-12)

    rows = list(csv.DictReader((output / "umma_device_scaling.csv").open()))
    configuration_rows = [r for r in rows if r["section"] == "configuration"]
    check("summary CSV covers all four configurations",
          len({(r["method"], r["scale"]) for r in configuration_rows}) == 4)
    sample_rows = [r for r in rows if r["metric"] == "within_campaign_sample_count"]
    check("within-campaign sample count is 30 and is not aggregated",
          all(r["campaign_1_value"] == "30" and r["mean"] == "not_applicable"
              for r in sample_rows), str(sample_rows[:1]))
    efficiency_rows = [r for r in rows if r["metric"] == "scaling_efficiency_1sm_percent"]
    check("scaling efficiency appears once as a derived metric",
          len(efficiency_rows) == 1 and efficiency_rows[0]["evidence_class"] == "derived")

    digests = dict(
        line.split("  ", 1)[::-1]
        for line in (output / "SHA256SUMS").read_text().splitlines()
    )
    check("SHA256SUMS covers every artifact",
          all(sha256_file(output / relative) == digest
              for relative, digest in ((k.strip(), v) for k, v in digests.items())))
    svg = (output / "figures" / "umma_device_scaling.svg").read_text()
    check("figure separates the isolated and device-scale scales",
          "Isolated work unit" in svg and "Device scale" in svg
          and "separate y-axes" in svg)
    check("figure shows scaling efficiency", "Scaling efficiency" in svg)


def case_missing_configuration(work: Path) -> None:
    root = work / "missing_configuration"
    campaigns = build_population(root)
    raw = campaigns[1] / "raw" / "umma_device_scaling.csv"
    rows = [r for r in csv.DictReader(raw.open())
            if not (r["method"] == "umma_2sm" and r["scale"] == "device_scale")]
    freeze(campaigns[1], rows, campaign_id=campaigns[1].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("missing configuration is rejected", root, campaigns,
                    "missing configuration(s)")


def case_duplicate_sample(work: Path) -> None:
    root = work / "duplicate_sample"
    campaigns = build_population(root)
    raw = campaigns[0] / "raw" / "umma_device_scaling.csv"
    rows = list(csv.DictReader(raw.open()))
    for row in rows:
        if row["method"] == "umma_1sm" and row["scale"] == "isolated" and row["sample_index"] == "7":
            row["sample_index"] = "6"
            break
    freeze(campaigns[0], rows, campaign_id=campaigns[0].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("duplicate sample is rejected", root, campaigns, "duplicate sample")


def case_wrong_sample_count(work: Path) -> None:
    root = work / "wrong_sample_count"
    campaigns = build_population(root)
    raw = campaigns[2] / "raw" / "umma_device_scaling.csv"
    rows = [r for r in csv.DictReader(raw.open()) if r["sample_index"] != "29"]
    freeze(campaigns[2], rows, campaign_id=campaigns[2].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("wrong sample count is rejected", root, campaigns,
                    "sample indexes are not exactly 0..29")


def case_invalid_experimental_contract(work: Path) -> None:
    cases = (
        ("fabricated throughput is rejected", "total_tflops", "20000.000000",
         "total_tflops does not match"),
        ("incorrect FLOP accounting is rejected", "total_flops", "1",
         "total_flops does not match"),
        ("incorrect matrix dimension is rejected", "m", "64", "m must be 256"),
        ("incorrect iteration count is rejected", "iterations", "1", "iterations must be 1000"),
        ("incorrect warm-up count is rejected", "warmup_iterations", "0",
         "warmup_iterations must be 10"),
        ("false full-device coverage is rejected", "coverage_status", "full_device_coverage",
         "coverage_status does not match active SM count"),
    )
    for index, (name, field, value, fragment) in enumerate(cases):
        root = work / f"invalid_contract_{index}"
        campaigns = build_population(root, sm_count=147 if field == "coverage_status" else 148)
        raw = campaigns[0] / "raw" / "umma_device_scaling.csv"
        rows = list(csv.DictReader(raw.open()))
        rows[3][field] = value
        freeze(campaigns[0], rows, campaign_id=campaigns[0].name, commit=COMMIT_A, gpu=GPU_A)
        expect_rejected(name, root, campaigns, fragment)

    root = work / "invalid_manifest_repetitions"
    campaigns = build_population(root)
    rows = [row for row in csv.DictReader((campaigns[0] / "raw" / "umma_device_scaling.csv").open())
            if row["sample_index"] == "0"]
    for row in rows:
        row["repetitions"] = "1"
    freeze(campaigns[0], rows, campaign_id=campaigns[0].name, commit=COMMIT_A, gpu=GPU_A)
    manifest_path = campaigns[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["repetitions"] = 1
    manifest_path.write_text(json.dumps(manifest))
    expect_rejected("one-repetition final campaign is rejected", root, campaigns, "frozen contract")

    root = work / "missing_source_provenance"
    campaigns = build_population(root)
    manifest_path = campaigns[0] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = {}
    manifest_path.write_text(json.dumps(manifest))
    expect_rejected("missing source provenance is rejected", root, campaigns, "source inventory")


def case_mixed_commit(work: Path) -> None:
    root = work / "mixed_commit"
    campaigns = build_population(root)
    rows = make_rows(commit=COMMIT_B, gpu=GPU_A, campaign_kind="final", sm_count=148, offset=0.002)
    freeze(campaigns[2], rows, campaign_id=campaigns[2].name, commit=COMMIT_B, gpu=GPU_A)
    expect_rejected("mixed Git commit is rejected", root, campaigns,
                    "do not share one Git commit")


def case_mixed_gpu(work: Path) -> None:
    root = work / "mixed_gpu"
    campaigns = build_population(root)
    rows = make_rows(commit=COMMIT_A, gpu=GPU_B, campaign_kind="final", sm_count=148, offset=0.002)
    freeze(campaigns[1], rows, campaign_id=campaigns[1].name, commit=COMMIT_A, gpu=GPU_B)
    expect_rejected("mixed GPU is rejected", root, campaigns, "do not share one GPU UUID")


def case_hash_mismatch(work: Path) -> None:
    root = work / "hash_mismatch"
    campaigns = build_population(root)
    manifest = json.loads((campaigns[0] / "manifest.json").read_text())
    manifest["artifact_sha256"]["raw/umma_device_scaling.csv"] = "e" * 64
    (campaigns[0] / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    expect_rejected("hash mismatch is rejected", root, campaigns, "hash mismatch")


def case_correctness_failure(work: Path) -> None:
    root = work / "correctness_failure"
    campaigns = build_population(root)
    raw = campaigns[1] / "raw" / "umma_device_scaling.csv"
    rows = list(csv.DictReader(raw.open()))
    rows[17]["correctness"] = "FAIL"
    rows[17]["mismatches"] = "4096"
    freeze(campaigns[1], rows, campaign_id=campaigns[1].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("correctness failure is rejected", root, campaigns, "correctness is not OK")


def case_incomplete_coverage(work: Path) -> None:
    root = work / "incomplete_coverage"
    campaigns = build_population(root)
    raw = campaigns[0] / "raw" / "umma_device_scaling.csv"
    rows = list(csv.DictReader(raw.open()))
    for row in rows:
        if row["scale"] == "device_scale":
            row["coverage_status"] = "incomplete_coverage"
    freeze(campaigns[0], rows, campaign_id=campaigns[0].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("incomplete device coverage is rejected", root, campaigns,
                    "cannot be published")


def case_unproven_residency(work: Path) -> None:
    root = work / "unproven_residency"
    campaigns = build_population(root)
    raw = campaigns[0] / "raw" / "umma_device_scaling.csv"
    rows = list(csv.DictReader(raw.open()))
    for row in rows:
        if row["scale"] == "device_scale":
            row["residency_evidence"] = "not_established"
    freeze(campaigns[0], rows, campaign_id=campaigns[0].name, commit=COMMIT_A, gpu=GPU_A)
    expect_rejected("unproven residency is rejected", root, campaigns,
                    "residency was not established")


def case_odd_sm_count(work: Path) -> None:
    """An odd SM count is a legitimate device, not a malformed campaign: the
    analyzer must accept it and expose the unequal active-SM coverage."""
    root = work / "odd_sm_count"
    campaigns = build_population(root, sm_count=147)
    output = root / "analysis"
    completed = run_analyzer(campaigns, output)
    check("odd SM count is accepted", completed.returncode == 0,
          f"rc={completed.returncode} stderr={completed.stderr.strip()[-500:]!r}")
    if completed.returncode != 0:
        return
    summary = json.loads((output / "summary.json").read_text())
    check("odd SM count reports unequal active-SM coverage",
          summary["equal_active_sm_coverage"] is False)
    warnings = " ".join(summary["warnings"])
    check("odd SM count warns about the unequal comparison",
          "not directly comparable" in warnings, warnings)
    check("odd SM count warns about the unused SM", "1 of 147 SMs unused" in warnings, warnings)
    check("odd SM count is not labelled full-device coverage",
          "maximum_resident_coverage" in warnings, warnings)
    derived = summary["per_campaign_derived"][0]
    check("odd SM count keeps the 1-SM arm at 147 SMs",
          derived["planned_active_sm_1sm"] == 147)
    check("odd SM count keeps the 2-SM arm at 146 SMs",
          derived["planned_active_sm_2sm"] == 146)
    rows = list(csv.DictReader((output / "umma_device_scaling.csv").open()))
    equal_rows = [r for r in rows if r["metric"] == "equal_active_sm_coverage"]
    check("summary CSV exposes the unequal coverage flag",
          len(equal_rows) == 1 and equal_rows[0]["campaign_1_value"] == "False")


def case_population_shape(work: Path) -> None:
    root = work / "population_shape"
    campaigns = build_population(root)
    expect_rejected("two campaigns are rejected", root, campaigns[:2],
                    "exactly three --campaign arguments are required")
    expect_rejected("a repeated campaign is rejected", root,
                    [campaigns[0], campaigns[0], campaigns[1]], "must be distinct")

    pilot = root / "20260824T150000Z"
    rows = make_rows(commit=COMMIT_A, gpu=GPU_A, campaign_kind="pilot", sm_count=148, offset=0.0)
    freeze(pilot, rows, campaign_id=pilot.name, commit=COMMIT_A, gpu=GPU_A, campaign_kind="pilot")
    expect_rejected("a pilot campaign is rejected", root, [campaigns[0], campaigns[1], pilot],
                    "only COMPLETE final campaigns are accepted")


def case_cli(work: Path) -> None:
    completed = subprocess.run([sys.executable, str(ANALYZER), "--help"], cwd=ROOT, text=True,
                               capture_output=True, check=False)
    check("--help exits zero", completed.returncode == 0)
    check("--help names the experiment", "device-scaling" in completed.stdout)
    completed = subprocess.run([sys.executable, str(ANALYZER)], cwd=ROOT, text=True,
                               capture_output=True, check=False)
    check("missing --output is a usage error", completed.returncode == 2)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="umma-scaling-tests-") as name:
        work = Path(name)
        for case in (case_valid, case_missing_configuration, case_duplicate_sample,
                     case_wrong_sample_count, case_invalid_experimental_contract,
                     case_mixed_commit, case_mixed_gpu,
                     case_hash_mismatch, case_correctness_failure, case_incomplete_coverage,
                     case_unproven_residency, case_odd_sm_count, case_population_shape,
                     case_cli):
            case(work)
    if FAILURES:
        print(f"\numma-scaling-tests: FAILED, {len(FAILURES)} check(s)", file=sys.stderr)
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}", file=sys.stderr)
        return 1
    print("\numma-scaling-tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
