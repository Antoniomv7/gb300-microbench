#!/usr/bin/env python3
"""Profile one BF16 GEMM launch per shape for persistent_2cta and cuBLASLt with Nsight Compute.

Each capture runs in its own process under ncu. That worker reuses the operand generation,
layouts, validation, kernel preparation and warm-up of gemm_comparison/gemm_comparison.py and
wraps exactly one further launch in the only NVTX range the Nsight Compute filter admits.
Profiler durations are diagnostics; the CUDA-event results of the analysis named by
--gemm-summary remain the performance data.
"""

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ncu_capture  # noqa: E402  (scripts/ is this script's directory)
import provenance  # noqa: E402
import run_campaign  # noqa: E402
from gemm_comparison import gemm_comparison as gemm  # noqa: E402

SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1), (32768, 512, 4096, 1))
VARIANTS = ("persistent_2cta", "heuristic_first_supported")
CAMPAIGN_WARMUP = run_campaign.GEMM_PROTOCOL["final"]["warmup"]
NVTX_RANGE = "gb300_gemm_profile"
DRAM_METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
TIMING_METRICS = ("gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second")
# Bytes returned from L2 to the SMs for global TMA loads; both GEMMs load operands with TMA.
L2_READ_BASE = "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld"
L2_READ_METRIC = f"{L2_READ_BASE}.sum"
L2_CALIBRATION = {"section": "memory_paths", "method": "tma", "stages": 4,
                  "bytes_in_flight_kib": 64, "kernel_name": "tma_benchmark_kernel"}
L2_CALIBRATION_TOLERANCE = 1e-3
# Cold is the original isolated-kernel diagnostic. Hot re-runs the deterministic
# application, including its warm-up, for every profiling pass; NCU does not flush L2.
# Both modes retain the campaign's unlocked clocks and dynamic pipeline boost.
NCU_SETTINGS = {
    "cold": (("--clock-control", "none"), ("--pipeline-boost-state", "dynamic"),
             ("--cache-control", "all"), ("--replay-mode", "kernel")),
    "hot": (("--clock-control", "none"), ("--pipeline-boost-state", "dynamic"),
            ("--cache-control", "none"), ("--replay-mode", "application")),
}
CAPTURE_TIMEOUT_S = 1800
SOURCES = ("scripts/profile_gemm.py", "scripts/provenance.py", "scripts/ncu_capture.py",
           "scripts/run_campaign.py", "gemm_comparison/gemm_comparison.py",
           "gemm_comparison/cublaslt_bridge.cu")
TIME_SCALE_NS = {"ns": 1.0, "nsecond": 1.0, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6}
CLOCK_SCALE_HZ = {"hz": 1.0, "cycle/second": 1.0, "cycle/nsecond": 1e9}
SUMMARY_FIELDS = ("shape_id", "m", "n", "k", "l", "variant", "method", "cache_state", "status",
                  "kernel_name",
                  "validation", "dram_read_bytes", "dram_write_bytes", "compulsory_read_bytes",
                  "dram_read_to_compulsory", "dram_read_excess_bytes", "output_bytes",
                  "dram_write_to_output", "l2_tma_read_bytes", "l2_tma_read_to_dram_read",
                  "profiled_duration_us", "profiled_sm_clock_mhz",
                  "cuda_event_mean_kernel_time_us", "cuda_event_mean_tflops")


def shape_id(shape):
    return "x".join(map(str, shape))


def parse_shape(value):
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return (*dimensions, 1)


def cublaslt_runtime():
    library = ctypes.CDLL("libcublasLt.so.13")
    library.cublasLtGetVersion.restype = ctypes.c_size_t
    return {"version": int(library.cublasLtGetVersion()),
            "libraries": provenance.loaded_library("libcublasLt")}


def digest(torch, tensor):
    data = tensor.detach().cpu().contiguous().flatten().view(torch.uint8)
    return hashlib.sha256(data.numpy()).hexdigest()


def worker(shape, variant, result_path):
    """Run inside ncu: prepare as the campaigns do, then launch once inside the NVTX range."""
    import cutlass
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver

    specification = next(item for item in gemm.CANDIDATES if item["variant"] == variant)
    modules = {"nonpersistent": gemm.load_module("dense_gemm", "dense_gemm.py"),
               "persistent": gemm.load_module("dense_gemm_persistent", "dense_gemm_persistent.py")}
    torch_stream = torch.cuda.current_stream()
    context = (torch, cutlass, cute, torch_stream, driver.CUstream(torch_stream.cuda_stream))
    operands = gemm.create_operands(torch, cutlass, modules["nonpersistent"], shape)
    reference = gemm.reference_result(torch, operands)
    # Compilation, the reference, the first validated launch and warm-up stay outside the range.
    label, launch, bridge = gemm.prepare_candidate(specification, modules, shape, operands,
                                                   reference, context, CAMPAIGN_WARMUP)
    try:
        torch.cuda.nvtx.range_push(NVTX_RANGE)
        launch()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        # The replayed launch must still leave the validated result in the output.
        gemm.validate_result(torch, operands["output"], reference, label)
        error = float((operands["output"] - reference).abs().max().item())
    finally:
        if bridge is not None:
            bridge.close()
    properties = torch.cuda.get_device_properties(0)
    record = {
        "label": label, "variant": variant, "method": specification["method"],
        "operands": {"a": "BFloat16 (M,K,L), K-major", "b": "BFloat16 (N,K,L), K-major",
                     "output": "Float32 (M,N,L), N-major", "accumulator": "Float32",
                     "seed": 1111, "generator": "gemm_comparison.create_operands",
                     "sha256": {"a": digest(torch, operands["a_gpu"]),
                                "b": digest(torch, operands["b_gpu"])}},
        "validation": {"reference": "IEEE-FP32 einsum of the BF16 operands "
                                    "(gemm_comparison.reference_result)",
                       "atol": gemm.ATOL, "rtol": gemm.RTOL,
                       "first_launch": "PASS", "after_profiled_launch": "PASS",
                       "max_abs_error_after_profiled_launch": error},
        "warmup_launches": CAMPAIGN_WARMUP, "launches_in_nvtx_range": 1, "nvtx_range": NVTX_RANGE,
        "device": {"name": properties.name, "sm_count": properties.multi_processor_count,
                   "l2_cache_bytes": properties.L2_cache_size},
        "cublaslt_runtime": cublaslt_runtime(),
    }
    Path(result_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def ncu(*arguments):
    return [os.environ.get("NCU_BINARY", "ncu"), *map(str, arguments)]


class Log:
    """Mirror progress messages to stderr and the study's execution log."""

    def __init__(self, path):
        self.handle = path.open("w", encoding="utf-8")

    def __call__(self, message):
        line = f"{dt.datetime.now(dt.timezone.utc).isoformat()} profile: {message}"
        print(line, file=sys.stderr, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, timeout=CAPTURE_TIMEOUT_S)
    return completed.returncode


def export_csv(report, destination):
    completed = subprocess.run(
        ncu("--import", report, "--csv", "--page", "raw", "--print-units", "base",
            "--print-fp", "--print-kernel-base", "function"),
        cwd=ROOT, text=True, capture_output=True, check=True, timeout=600)
    destination.write_text(completed.stdout, encoding="utf-8")
    return completed.stdout


def parse_kernels(text, metrics):
    """Return one record per profiled kernel from a raw-page NCU CSV export."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("empty NCU export")
    header = [field.strip().lstrip("﻿") for field in rows[0]]
    missing = [name for name in ("Kernel Name", *metrics) if name not in header]
    if missing:
        raise ValueError(f"NCU export lacks {missing}")
    units = {metric: rows[1][header.index(metric)].strip() for metric in metrics}
    nvtx = next((index for index, name in enumerate(header) if "Push/Pop_Range" in name), None)
    kernels = []
    for row in rows[2:]:
        column = dict(zip(header, row))
        kernels.append({
            "name": column["Kernel Name"], "block_size": column.get("Block Size", ""),
            "grid_size": column.get("Grid Size", ""),
            "nvtx_ranges": row[nvtx].strip() if nvtx is not None else "",
            "metrics": {metric: float(column[metric].replace(",", "")) for metric in metrics}})
    return kernels, units


def query_metric(base):
    """Return the device's own description of base, or None when this GPU lacks it."""
    output = subprocess.run(ncu("--devices", "0", "--query-metrics"), cwd=ROOT, text=True,
                            capture_output=True, check=True, timeout=600).stdout
    chip = next((line.strip() for line in output.splitlines() if line.startswith("Device ")), "")
    for line in output.splitlines():
        fields = re.split(r"\s{2,}", line.strip())
        if fields[0] == base and len(fields) >= 4:
            return {"name": base, "type": fields[1], "unit": fields[2],
                    "description": fields[3], "device": chip}
    return None


def calibrate_l2_metric(directory, log, ncu_settings):
    """Admit the L2 read metric only if the device supports it and a known stream confirms it."""
    query = query_metric(L2_READ_BASE)
    record = {"metric": L2_READ_METRIC, "query": query, "collected": False}
    if query is None:
        record["reason"] = "not reported by ncu --query-metrics for this device"
        return record
    # A TMA stream with no reuse reads every logical byte from L2 exactly once.
    case = directory / "calibration_tma_stream"
    metrics = (*DRAM_METRICS, L2_READ_METRIC)
    command = ncu(*(item for pair in ncu_settings for item in pair), "--devices", "0",
                  "--kernel-name-base", "function", "--kernel-name", L2_CALIBRATION["kernel_name"],
                  "--launch-count", "1", "--print-summary", "none",
                  "--metrics", ",".join(metrics), "-o", case, "--",
                  *ncu_capture.benchmark_command(L2_CALIBRATION))
    log(f"calibration: {' '.join(command)}")
    status = run_logged(command, case.with_suffix(".log"))
    record.update({"report": f"{case.name}.ncu-rep", "csv": f"{case.name}.csv",
                   "log": f"{case.name}.log", "benchmark": L2_CALIBRATION})
    if status:
        record["reason"] = f"calibration capture exited with status {status}"
        return record
    kernels, units = parse_kernels(export_csv(case.with_suffix(".ncu-rep"),
                                              case.with_suffix(".csv")), metrics)
    # Profiler messages can interleave with the benchmark's own CSV lines.
    lines = case.with_suffix(".log").read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith(("method,", "tma,"))]
    useful = float(next(csv.DictReader(rows))["useful_bytes"])
    values = kernels[0]["metrics"]
    ratio = values[L2_READ_METRIC] / useful
    record.update({"useful_bytes": useful, "metrics": values, "units": units,
                   "l2_read_to_useful": ratio,
                   "dram_read_to_useful": values["dram__bytes_read.sum"] / useful,
                   "tolerance": L2_CALIBRATION_TOLERANCE})
    if len(kernels) == 1 and abs(ratio - 1) <= L2_CALIBRATION_TOLERANCE:
        record["collected"] = True
    else:
        record["reason"] = f"calibration ratio {ratio:.6f} outside ±{L2_CALIBRATION_TOLERANCE}"
    return record


def resolve(path):
    return path if path.is_absolute() else ROOT / path


def read_gemm_summary(path, gpu):
    """Load the CUDA-event reference from an analysis whose manifest names this GPU.

    The published results/ predate the manifest, so a new profile cannot fall back to them.
    """
    path = resolve(path)
    manifest_path = path.parent / "analysis.json"
    if path.name != "gemm_comparison.csv" or not path.exists() or not manifest_path.exists():
        raise ValueError(f"{path} must be the gemm_comparison.csv of a make analyze output "
                         "directory, next to its analysis.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("outputs", {}).get(path.name) != digest:
        raise ValueError(f"{path} differs from the file recorded in {manifest_path}")
    if manifest.get("gpu_uuid") != gpu["uuid"]:
        raise ValueError(f"{path} was measured on {manifest.get('gpu_uuid')}, not on the "
                         f"profiled GPU {gpu['uuid']}")
    with path.open(newline="", encoding="utf-8") as source:
        rows = {(row["shape_id"], row["variant"]): row for row in csv.DictReader(source)}
    missing = [f"{shape_id(shape)}/{variant}" for shape in SHAPES for variant in VARIANTS
               if (shape_id(shape), variant) not in rows]
    if missing:
        raise ValueError(f"{path} lacks the rows {missing}")
    return {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "sha256": digest, "manifest": "analysis.json",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "source_commit": manifest["source_commit"], "gpu_uuid": manifest["gpu_uuid"],
            "campaigns": [campaign["campaign_id"] for campaign in manifest["campaigns"]],
            "rows": {f"{shape_id(shape)}/{variant}": {
                "mean_tflops": float(rows[(shape_id(shape), variant)]["mean_tflops"]),
                "mean_kernel_time_us":
                    1e3 * float(rows[(shape_id(shape), variant)]["mean_kernel_time_ms"])}
                for shape in SHAPES for variant in VARIANTS}}


def traffic(shape, metrics, units):
    """Relate the aggregate counters to the compulsory operand and output bytes."""
    m, n, k, batch = shape
    a_bytes, b_bytes, output_bytes = 2 * m * k * batch, 2 * n * k * batch, 4 * m * n * batch
    compulsory = a_bytes + b_bytes
    read, write = metrics["dram__bytes_read.sum"], metrics["dram__bytes_write.sum"]
    duration_ns = metrics["gpu__time_duration.sum"] * TIME_SCALE_NS[
        units["gpu__time_duration.sum"].lower()]
    clock_hz = metrics["sm__cycles_elapsed.avg.per_second"] * CLOCK_SCALE_HZ[
        units["sm__cycles_elapsed.avg.per_second"].lower()]
    result = {"a_bytes": a_bytes, "b_bytes": b_bytes, "output_bytes": output_bytes,
              "compulsory_read_bytes": compulsory,
              "dram_read_to_compulsory": read / compulsory,
              "dram_read_excess_bytes": read - compulsory,
              # Hypothetical equivalents only: aggregate counters cannot attribute rereads.
              "dram_read_excess_in_a_sizes": (read - compulsory) / a_bytes,
              "dram_read_excess_in_b_sizes": (read - compulsory) / b_bytes,
              "dram_write_to_output": write / output_bytes,
              "profiled_duration_us": duration_ns / 1e3,
              "profiled_sm_clock_mhz": clock_hz / 1e6,
              "profiled_dram_read_tb_per_s": read / duration_ns / 1e3}
    if L2_READ_METRIC in metrics:
        result["l2_tma_read_bytes"] = metrics[L2_READ_METRIC]
        result["l2_tma_read_to_dram_read"] = metrics[L2_READ_METRIC] / read
    return result


def capture(directory, index, shape, variant, metrics, log, ncu_settings, reference):
    case = f"{index:02d}_{shape_id(shape[:3])}_{variant}"
    stem = directory / case
    command = ncu(*(item for pair in ncu_settings for item in pair), "--devices", "0",
                  "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/", "--kernel-name-base", "function",
                  "--print-summary", "none", "--metrics", ",".join(metrics), "-o", stem, "--",
                  sys.executable, Path(__file__).resolve(), "--worker",
                  "--shape", ",".join(map(str, shape[:3])), "--variant", variant,
                  "--result", stem.with_suffix(".worker.json"))
    log(f"{case}: {' '.join(command)}")
    record = {"case": case, "shape_id": shape_id(shape), "m": shape[0], "n": shape[1],
              "k": shape[2], "l": shape[3], "variant": variant,
              "method": "cutedsl" if variant == "persistent_2cta" else "cublaslt",
              "files": {"report": f"{case}.ncu-rep", "csv": f"{case}.csv", "log": f"{case}.log",
                        "worker": f"{case}.worker.json"},
              "status": "FAIL", "problems": []}
    status = run_logged(command, stem.with_suffix(".log"))
    if status:
        record["problems"].append(f"ncu or the worker exited with status {status}")
    worker_path = stem.with_suffix(".worker.json")
    if worker_path.exists():
        record["worker"] = json.loads(worker_path.read_text(encoding="utf-8"))
        validation = record["worker"]["validation"]
        record["validation"] = ("PASS" if validation["first_launch"] == "PASS" and
                                validation["after_profiled_launch"] == "PASS" else "FAIL")
    else:
        record["validation"] = "FAIL"
        record["problems"].append("the worker did not report a validated launch")
    report = stem.with_suffix(".ncu-rep")
    if report.exists():
        try:
            kernels, units = parse_kernels(export_csv(report, stem.with_suffix(".csv")), metrics)
        except (ValueError, subprocess.SubprocessError) as error:
            kernels, units = [], {}
            record["problems"].append(f"cannot read the report: {error}")
        record["kernel_count"] = len(kernels)
        record["units"] = units
        if len(kernels) != 1:
            record["problems"].append(f"expected one kernel in the NVTX range, found {len(kernels)}")
        else:
            kernel = kernels[0]
            record["kernel"] = {key: kernel[key] for key in ("name", "block_size", "grid_size",
                                                             "nvtx_ranges")}
            record["metrics"] = kernel["metrics"]
            if NVTX_RANGE not in kernel["nvtx_ranges"]:
                record["problems"].append("the profiled kernel lies outside the NVTX range")
            if not kernel["name"]:
                record["problems"].append("the export has no kernel name")
            if any(not math.isfinite(value) or value < 0 for value in kernel["metrics"].values()):
                record["problems"].append("a requested metric is missing or negative")
            else:
                record["traffic"] = traffic(shape, kernel["metrics"], units)
    else:
        record["problems"].append("ncu wrote no report")
    record["cuda_event_reference"] = reference
    if not record["problems"] and record["validation"] == "PASS":
        record["status"] = "PASS"
    log(f"{case}: {record['status']} {record['problems'] or ''}".rstrip())
    return record


def summary_rows(captures, cache_state):
    rows = []
    for record in captures:
        traffic_record = record.get("traffic", {})
        reference = record.get("cuda_event_reference") or {}
        metrics = record.get("metrics", {})
        rows.append({
            **{key: record[key] for key in ("shape_id", "m", "n", "k", "l", "variant",
                                            "method", "status", "validation")},
            "cache_state": cache_state,
            "kernel_name": record.get("kernel", {}).get("name", ""),
            "dram_read_bytes": metrics.get("dram__bytes_read.sum", ""),
            "dram_write_bytes": metrics.get("dram__bytes_write.sum", ""),
            "output_bytes": traffic_record.get("output_bytes", ""),
            "l2_tma_read_bytes": traffic_record.get("l2_tma_read_bytes", ""),
            **{key: traffic_record.get(key, "") for key in (
                "compulsory_read_bytes", "dram_read_to_compulsory", "dram_read_excess_bytes",
                "dram_write_to_output", "l2_tma_read_to_dram_read", "profiled_duration_us",
                "profiled_sm_clock_mhz")},
            "cuda_event_mean_kernel_time_us": reference.get("mean_kernel_time_us", ""),
            "cuda_event_mean_tflops": reference.get("mean_tflops", "")})
    return rows


def profile(output, cache_state, gemm_summary):
    ncu_settings = NCU_SETTINGS[cache_state]
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    environment = {"gpu": provenance.gpu_identity(), "software": provenance.software_versions(),
                   "ncu": provenance.ncu_version(),
                   "repository": provenance.repository_state(SOURCES),
                   "pinned": provenance.pinned_versions()}
    # Reject a missing, altered or foreign CUDA-event reference before any capture.
    summary = read_gemm_summary(gemm_summary, environment["gpu"])
    directory = resolve(output)
    directory.mkdir(parents=True, exist_ok=False)
    log = Log(directory / "profile.log")
    log(f"GPU {environment['gpu']['uuid']} ({environment['gpu']['name']}), {environment['ncu']}")
    log(f"CUDA-event reference {summary['path']} (sha256 {summary['sha256']}; campaigns "
        f"{', '.join(summary['campaigns'])}; commit {summary['source_commit']})")

    l2_metric = calibrate_l2_metric(directory, log, ncu_settings)
    log(f"L2 read metric {L2_READ_METRIC}: "
        f"{'collected' if l2_metric['collected'] else 'not collected: ' + l2_metric['reason']}")
    metrics = (*DRAM_METRICS, *TIMING_METRICS, *((L2_READ_METRIC,) if l2_metric["collected"] else ()))

    captures = [capture(directory, index, shape, variant, metrics, log, ncu_settings,
                        summary["rows"][f"{shape_id(shape)}/{variant}"])
                for index, (shape, variant) in enumerate(
                    (shape, variant) for shape in SHAPES for variant in VARIANTS)]
    passed = sum(record["status"] == "PASS" for record in captures)
    expected = len(SHAPES) * len(VARIANTS)
    digests = {record["shape_id"]: set() for record in captures}
    for record in captures:
        if "worker" in record:
            digests[record["shape_id"]].add(json.dumps(record["worker"]["operands"]["sha256"]))
    shared_operands = all(len(values) == 1 for values in digests.values())
    complete = passed == expected == len(captures) and shared_operands
    with (directory / "gemm_profile.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows(captures, cache_state))
    index = {
        "study": "gemm_profile", "cache_state": cache_state,
        "state": "COMPLETE" if complete else "INCOMPLETE",
        "created_utc": created, "command": [sys.executable, *sys.argv],
        "expected_count": expected, "captured_count": len(captures), "passed_count": passed,
        "shared_operands_per_shape": shared_operands,
        "environment": environment,
        "gemm_summary": {key: value for key, value in summary.items() if key != "rows"},
        "protocol": {
            "shapes": [shape_id(shape) for shape in SHAPES], "variants": list(VARIANTS),
            "operands_and_validation": "gemm_comparison.create_operands, reference_result, "
                                       "validate_result and prepare_candidate",
            "warmup_launches": CAMPAIGN_WARMUP, "launches_in_nvtx_range": 1,
            "outside_capture": ["compilation or plan creation", "reference calculation",
                                "first validated launch", "warm-up launches",
                                "post-capture validation"],
            "nvtx_filter": f"{NVTX_RANGE}/",
            "ncu_settings": {flag.lstrip("-"): value for flag, value in ncu_settings},
            "cache_interpretation": ("profiler flushes caches before replay passes" if
                                     cache_state == "cold" else
                                     "application replays validation and warm-up before each "
                                     "profiled launch; profiler does not flush caches"),
            "metrics": list(metrics),
            "performance_results": f"CUDA-event measurements in {summary['path']}; "
                                   "profiler durations are diagnostics"},
        "l2_read_metric": l2_metric,
        "captures": captures,
    }
    (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    log(f"{index['state']}: {passed}/{expected} captures passed; operands shared per shape: "
        f"{shared_operands}; {directory}")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-state", choices=tuple(NCU_SETTINGS),
                        help="cold: kernel replay after a cache flush; hot: application replay "
                             "of validation and warm-up before every pass, without a flush")
    parser.add_argument("--gemm-summary", type=Path,
                        help="gemm_comparison.csv written by make analyze for the new campaigns")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", type=parse_shape, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not (args.shape and args.variant and args.result):
            parser.error("--worker requires --shape, --variant and --result")
        worker(args.shape, args.variant, args.result)
        return
    if not (args.output and args.cache_state and args.gemm_summary):
        parser.error("--output, --cache-state and --gemm-summary are required")
    if not profile(args.output, args.cache_state, args.gemm_summary):
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"profile: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
