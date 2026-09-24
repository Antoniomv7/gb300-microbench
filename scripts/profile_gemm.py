#!/usr/bin/env python3
"""Profile one BF16 GEMM launch per shape for persistent_2cta and cuBLASLt with Nsight Compute.

Each capture runs in its own process under ncu. That worker reuses the operand generation,
layouts, validation, kernel preparation and warm-up of gemm_comparison/gemm_comparison.py and
wraps exactly one further launch in the only NVTX range the Nsight Compute filter admits.
Profiler durations are diagnostics; the CUDA-event results of Experiment IV remain the
performance data. analysis/analyze.py turns the exported counters into gemm_profile.csv.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gemm_comparison import gemm_comparison as gemm  # noqa: E402
from scripts import metadata, run_campaign  # noqa: E402
from scripts.ncu_capture import benchmark_command, export_csv, ncu, parse_kernels  # noqa: E402

SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1), (32768, 512, 4096, 1))
VARIANTS = ("persistent_2cta", "heuristic_first_supported")
CAMPAIGN_WARMUP = run_campaign.GEMM["warmup_iterations"]
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


def shape_id(shape):
    return "x".join(map(str, shape))


# Capture order and file stems: both implementations of each shape.
CASES = [(f"{index:02d}_{shape_id(shape[:3])}_{variant}", shape, variant)
         for index, (shape, variant) in enumerate(
             (shape, variant) for shape in SHAPES for variant in VARIANTS)]


def parse_shape(value):
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return (*dimensions, 1)


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
    record = {
        "label": label, "variant": variant, "method": specification["method"],
        # profile() requires both implementations of a shape to report the same operands.
        "operands_sha256": {"a": digest(torch, operands["a_gpu"]),
                            "b": digest(torch, operands["b_gpu"])},
        "validation": {"atol": gemm.ATOL, "rtol": gemm.RTOL,
                       "first_launch": "PASS", "after_profiled_launch": "PASS",
                       "max_abs_error_after_profiled_launch": error},
    }
    Path(result_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_logged(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, timeout=CAPTURE_TIMEOUT_S)
    return completed.returncode


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


def calibrate_l2_metric(directory, ncu_settings):
    """Admit the L2 read metric only if the device supports it and a known stream confirms it."""
    query = query_metric(L2_READ_BASE)
    record = {"metric": L2_READ_METRIC, "query": query, "collected": False,
              "benchmark": L2_CALIBRATION, "tolerance": L2_CALIBRATION_TOLERANCE}
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
                  *benchmark_command(L2_CALIBRATION))
    print(f"profile: calibration: {' '.join(command)}", file=sys.stderr, flush=True)
    status = run_logged(command, case.with_suffix(".log"))
    if status:
        record["reason"] = f"calibration capture exited with status {status}"
        return record
    kernels, units = parse_kernels(export_csv(case.with_suffix(".ncu-rep"),
                                              case.with_suffix(".csv")), metrics)
    if len(kernels) != 1:
        record["reason"] = f"calibration captured {len(kernels)} kernels; expected one"
        return record
    # Profiler messages can interleave with the benchmark's own CSV lines.
    lines = case.with_suffix(".log").read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith(("method,", "tma,"))]
    useful = float(next(csv.DictReader(rows))["useful_bytes"])
    values = kernels[0]["metrics"]
    ratio = values[L2_READ_METRIC] / useful
    record.update({"useful_bytes": useful, "metrics": values, "units": units,
                   "l2_read_to_useful": ratio,
                   "dram_read_to_useful": values["dram__bytes_read.sum"] / useful})
    if abs(ratio - 1) <= L2_CALIBRATION_TOLERANCE:
        record["collected"] = True
    else:
        record["reason"] = f"calibration ratio {ratio:.6f} outside ±{L2_CALIBRATION_TOLERANCE}"
    return record


def capture(directory, case, shape, variant, metrics, ncu_settings):
    """Profile one validated launch; fail unless exactly one named kernel ran in the range."""
    stem = directory / case
    command = ncu(*(item for pair in ncu_settings for item in pair), "--devices", "0",
                  "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/", "--kernel-name-base", "function",
                  "--print-summary", "none", "--metrics", ",".join(metrics), "-o", stem, "--",
                  sys.executable, Path(__file__).resolve(), "--worker",
                  "--shape", ",".join(map(str, shape[:3])), "--variant", variant,
                  "--result", stem.with_suffix(".worker.json"))
    print(f"profile: {case}: {' '.join(map(str, command))}", file=sys.stderr, flush=True)
    status = run_logged(command, stem.with_suffix(".log"))
    if status:
        raise RuntimeError(f"{case}: ncu or the validated worker exited with status {status}; "
                           f"see {stem.with_suffix('.log')}")
    # The worker writes its record only after both validations passed.
    record = json.loads(stem.with_suffix(".worker.json").read_text(encoding="utf-8"))
    kernels, _ = parse_kernels(export_csv(stem.with_suffix(".ncu-rep"),
                                          stem.with_suffix(".csv")), metrics)
    if len(kernels) != 1 or NVTX_RANGE not in kernels[0]["nvtx_ranges"] or not kernels[0]["name"]:
        raise RuntimeError(f"{case}: expected one named kernel inside the NVTX range, "
                           f"found {len(kernels)}")
    values = kernels[0]["metrics"]
    if any(not math.isfinite(value) or value < 0 for value in values.values()) or \
            any(values[metric] <= 0 for metric in TIMING_METRICS):
        raise RuntimeError(f"{case}: a counter is invalid, or the duration or SM clock is zero")
    return record


def profile(output, cache_state):
    ncu_settings = NCU_SETTINGS[cache_state]
    directory = output if output.is_absolute() else ROOT / output
    directory.mkdir(parents=True, exist_ok=False)
    record = {"cache_state": cache_state, "created_utc": metadata.utc_now(),
              **metadata.environment(),
              "ncu_settings": {flag.lstrip("-"): value for flag, value in ncu_settings},
              "nvtx_range": NVTX_RANGE, "warmup_launches": CAMPAIGN_WARMUP}

    calibration = calibrate_l2_metric(directory, ncu_settings)
    verdict = "collected" if calibration["collected"] else f"not collected: {calibration['reason']}"
    print(f"profile: L2 read metric {L2_READ_METRIC}: {verdict}", file=sys.stderr, flush=True)
    metrics = (*DRAM_METRICS, *TIMING_METRICS,
               *((L2_READ_METRIC,) if calibration["collected"] else ()))

    operands = defaultdict(set)
    for case, shape, variant in CASES:
        worker_record = capture(directory, case, shape, variant, metrics, ncu_settings)
        operands[shape].add(json.dumps(worker_record["operands_sha256"], sort_keys=True))
    if any(len(digests) != 1 for digests in operands.values()):
        raise RuntimeError("the two implementations of a shape were profiled on different operands")

    record.update({"metrics": list(metrics), "l2_read_metric": calibration,
                   "captures": [case for case, _, _ in CASES]})
    metadata.complete(record)
    (directory / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"profile: complete {directory}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="new profile directory")
    parser.add_argument("--cache-state", choices=tuple(NCU_SETTINGS),
                        help="hot: application replay of validation and warm-up before every "
                             "pass, without a flush; cold: kernel replay after a cache flush")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", type=parse_shape, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.shape, args.variant, args.result)
    elif args.output and args.cache_state:
        profile(args.output, args.cache_state)
    else:
        parser.error("--output and --cache-state are required")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"profile: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
