#!/usr/bin/env python3
"""Check the GEMM profile and the extended precision comparison from their saved files.

The checks are deliberately independent of the scripts' own verdicts: they recount captures
and rows, re-read the exported Nsight Compute CSVs, and require every validation record to pass.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import profile_gemm  # noqa: E402  (scripts/ is this script's directory)
import provenance  # noqa: E402
from precision_comparison import precision_comparison as precision  # noqa: E402

IMPLEMENTATIONS = ("cutedsl", "cublaslt")
REPETITIONS = 3
OPERAND_SETS = ("validated", *(f"timed_{index}" for index in range(1, REPETITIONS + 1)))
COMPARISON_REQUIRED = (
    "shape_id", "precision", "implementation", "comparison", "input_dtype", "accumulator_dtype",
    "output_dtype", "alpha", "beta", "kernel", "repetition_1_tflops", "repetition_2_tflops",
    "repetition_3_tflops", "mean_tflops", "stdev_tflops", "cv_percent", "mean_kernel_time_us",
    "throughput_ratio_vs_cublaslt", "validation")
FP32_OUTPUTS = {"cutedsl": "Float32", "cublaslt": "CUDA_R_32F"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def check_provenance(environment):
    """A recorded GPU and commit, with no tracked source modified at acquisition time."""
    repository = environment.get("repository", {})
    problems = [] if repository.get("commit") and environment.get("gpu", {}).get("uuid") else \
        ["the GPU or the source commit was not recorded"]
    if provenance.tracked_changes(repository):
        problems.append("acquired with modified tracked files "
                        f"{provenance.tracked_changes(repository)}")
    return problems


def check_gemm_summary(index, captures):
    """The CUDA-event reference must be the manifest-backed analysis the profile recorded."""
    summary = index.get("gemm_summary", {})
    if not all(summary.get(key) for key in ("path", "sha256", "source_commit", "gpu_uuid",
                                            "campaigns")):
        return ["the CUDA-event reference (GEMM_SUMMARY) is not recorded"]
    problems = []
    path = profile_gemm.resolve(Path(summary["path"]))
    if not path.exists():
        problems.append(f"the CUDA-event reference {path} is missing")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != summary["sha256"]:
        problems.append(f"the CUDA-event reference {path} changed after profiling")
    if summary["gpu_uuid"] != index.get("environment", {}).get("gpu", {}).get("uuid"):
        problems.append("the CUDA-event reference was measured on a different GPU")
    for capture in captures:
        reference = capture.get("cuda_event_reference") or {}
        if not (positive(reference.get("mean_tflops")) and
                positive(reference.get("mean_kernel_time_us"))):
            problems.append(f"{capture.get('case', '?')}: no CUDA-event reference values")
    return problems


def check_gemm_profile(directory):
    """Six passing captures, each with its files, one kernel in the NVTX range and its metrics."""
    index_path = directory / "index.json"
    if not index_path.exists():
        return [f"{index_path} is missing"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    problems = [] if index.get("state") == "COMPLETE" else [f"state is {index.get('state')}"]
    problems += check_provenance(index.get("environment", {}))
    cache_state = index.get("cache_state")
    settings = {flag.lstrip("-"): value
                for flag, value in profile_gemm.NCU_SETTINGS.get(cache_state, ())}
    if not settings or index.get("protocol", {}).get("ncu_settings") != settings:
        problems.append(f"cache state {cache_state!r} and its Nsight Compute settings are not "
                        "recorded as defined")
    captures = index.get("captures", [])
    expected = {(profile_gemm.shape_id(shape), variant)
                for shape in profile_gemm.SHAPES for variant in profile_gemm.VARIANTS}
    found = [(capture.get("shape_id"), capture.get("variant")) for capture in captures]
    if len(found) != len(expected) or set(found) != expected:
        problems.append(f"{len(found)} captures cover {sorted(set(found))}; expected the "
                        f"{len(expected)} shape and implementation pairs")
    metrics = index.get("protocol", {}).get("metrics", [])
    missing = [metric for metric in (*profile_gemm.DRAM_METRICS, *profile_gemm.TIMING_METRICS)
               if metric not in metrics]
    if missing:
        problems.append(f"required metrics were not collected: {missing}")
    calibration = index.get("l2_read_metric", {})
    if (profile_gemm.L2_READ_METRIC in metrics) != bool(calibration.get("collected")):
        problems.append("the L2 read metric was collected without a passing calibration")
    if calibration.get("collected") and abs(calibration.get("l2_read_to_useful", 0) - 1) > \
            profile_gemm.L2_CALIBRATION_TOLERANCE:
        problems.append("the L2 read metric calibration is outside its tolerance")

    problems += check_gemm_summary(index, captures)
    for capture in captures:
        name = capture.get("case", "?")
        for kind, file in capture.get("files", {}).items():
            if not (directory / file).exists():
                problems.append(f"{name}: {kind} file {file} is missing")
        worker = capture.get("worker", {}).get("validation", {})
        if capture.get("status") != "PASS" or capture.get("validation") != "PASS" or \
                worker.get("first_launch") != "PASS" or worker.get("after_profiled_launch") != "PASS":
            problems.append(f"{name}: capture or numerical validation did not pass")
        export = directory / capture.get("files", {}).get("csv", "missing.csv")
        if not export.exists():
            continue
        try:
            kernels, _ = profile_gemm.parse_kernels(export.read_text(encoding="utf-8"), metrics)
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue
        if len(kernels) != 1:
            problems.append(f"{name}: the export holds {len(kernels)} kernels, expected one")
            continue
        kernel = kernels[0]
        if profile_gemm.NVTX_RANGE not in kernel["nvtx_ranges"] or not kernel["name"]:
            problems.append(f"{name}: the kernel is unnamed or outside the NVTX range")
        if kernel["name"] != capture.get("kernel", {}).get("name") or \
                kernel["metrics"] != capture.get("metrics"):
            problems.append(f"{name}: index.json disagrees with the exported CSV")
        if not all(math.isfinite(value) and value >= 0 for value in kernel["metrics"].values()) \
                or not kernel["metrics"].get("dram__bytes_read.sum", 0) > 0:
            problems.append(f"{name}: a metric is missing, negative or zero")

    summary = directory / "gemm_profile.csv"
    rows = read_rows(summary) if summary.exists() else []
    if len(rows) != len(expected) or any(row["status"] != "PASS" or row["validation"] != "PASS"
                                         or not row["kernel_name"] for row in rows):
        problems.append(f"{summary.name} does not hold {len(expected)} passing rows")
    if any(row.get("cache_state") != cache_state for row in rows):
        problems.append(f"{summary.name} mixes cache states or lacks them")
    return problems


def check_precision(directory):
    """Complete within-format comparison rows, all repetitions and passing validation."""
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return [f"{metadata_path} is missing"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    problems = [] if metadata.get("state") == "COMPLETE" else [f"state is {metadata.get('state')}"]
    problems += check_provenance(metadata.get("environment", {}))
    protocol = metadata.get("protocol", {})
    fixed = {"repetitions": REPETITIONS, "warmup_iterations": precision.WARMUP_ITERATIONS,
             "iterations": precision.ITERATIONS}
    if any(protocol.get(key) != value for key, value in fixed.items()):
        problems.append(f"the protocol is not {fixed}")
    configurations = {("x".join(map(str, shape)), name)
                      for shape in precision.SHAPES for name in precision.FORMATS}
    declared = {(shape, name) for shape in protocol.get("shapes", [])
                for name in protocol.get("formats", [])}
    if declared != configurations:
        problems.append("the run did not cover the three thesis shapes in all three formats")
    files = {name: directory / name for name in (
        "precision_cutedsl_vs_cublaslt.csv", "precision_cutedsl_vs_cublaslt.svg",
        "precision_comparison.csv", "precision_comparison.svg", "raw/repetitions.csv",
        "raw/validation.csv", "raw/operands.csv", "raw/cublaslt_plans.json")}
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        return problems + [f"missing files: {missing}"]

    comparison = read_rows(files["precision_cutedsl_vs_cublaslt.csv"])
    keys = [(row["shape_id"], row["precision"], row["implementation"]) for row in comparison]
    expected = {(shape, name, implementation) for shape, name in configurations
                for implementation in IMPLEMENTATIONS}
    if len(keys) != len(expected) or set(keys) != expected:
        problems.append(f"the comparison CSV has {len(keys)} rows; expected {len(expected)}")
    means = {key: float(row["mean_tflops"]) for key, row in zip(keys, comparison)
             if positive(row.get("mean_tflops"))}
    for key, row in zip(keys, comparison):
        label = "/".join(key)
        empty = [field for field in COMPARISON_REQUIRED if not row.get(field, "")]
        if empty:
            problems.append(f"{label}: empty fields {empty}")
        if row["validation"] != "PASS" or row["comparison"] != "matched":
            problems.append(f"{label}: validation {row['validation']}, {row['comparison']}")
        if row["output_dtype"] != FP32_OUTPUTS.get(key[2]):
            problems.append(f"{label}: output type {row['output_dtype']} is not FP32")
        if key[1] == "nvfp4" and (not row["scale_dtype"] or row["scale_block_elements"] != "16"):
            problems.append(f"{label}: NVFP4 scale type or block size is missing")
        if not all(positive(row.get(f"repetition_{index}_tflops"))
                   for index in range(1, REPETITIONS + 1)):
            problems.append(f"{label}: a repetition throughput is missing")
        baseline = means.get((key[0], key[1], "cublaslt"))
        if baseline and key in means and positive(row.get("throughput_ratio_vs_cublaslt")) and \
                abs(float(row["throughput_ratio_vs_cublaslt"]) - means[key] / baseline) > 1e-5:
            problems.append(f"{label}: the ratio does not match the two means")

    repetitions = read_rows(files["raw/repetitions.csv"])
    found = [(row["shape_id"], row["precision"], row["implementation"], row["repetition"])
             for row in repetitions]
    wanted = {(*key, str(index)) for key in expected for index in range(1, REPETITIONS + 1)}
    if len(found) != len(wanted) or set(found) != wanted:
        problems.append(f"raw/repetitions.csv has {len(found)} rows; expected {len(wanted)}")
    by_key = {(row["shape_id"], row["precision"], row["implementation"]): row for row in comparison}
    for key, row in zip(found, repetitions):
        summary = by_key.get(key[:3], {})
        if row["validation"] != "PASS" or not positive(row["kernel_time_us"]) or \
                row["tflops"] != summary.get(f"repetition_{row['repetition']}_tflops"):
            problems.append(f"{'/'.join(key)}: repetition is invalid or disagrees with the "
                            "comparison CSV")

    validation = read_rows(files["raw/validation.csv"])
    failures = [row for row in validation if row["status"] != "PASS"]
    if failures:
        problems.append(f"{len(failures)} validation records failed")
    stages = {(row["shape_id"], row["precision"], row["operand_set"], row["implementation"],
               row["stage"]) for row in validation}
    for shape, name in configurations:
        needed = {(shape, name, operand_set, "cutedsl", "after_example_run")
                  for operand_set in OPERAND_SETS}
        needed |= {(shape, name, operand_set, "cublaslt", "before_timing")
                   for operand_set in OPERAND_SETS}
        needed |= {(shape, name, operand_set, "cublaslt", "after_timing")
                   for operand_set in OPERAND_SETS[1:]}
        if needed - stages:
            problems.append(f"{shape}/{name}: {len(needed - stages)} validation records "
                            "are missing")

    operands = read_rows(files["raw/operands.csv"])
    if len(operands) != len(OPERAND_SETS) * len(configurations) or \
            any(row["represented_exactly"] != "True" for row in operands):
        problems.append("raw/operands.csv is incomplete or an operand was not represented exactly")
    # cuBLASLt must consume the very bytes, and NVFP4 scale bytes, that CuTe DSL consumed.
    identical = ("a_bytes_identical_to_cutedsl", "b_bytes_identical_to_cutedsl")
    scales = ("a_scale_bytes_identical_to_cutedsl", "b_scale_bytes_identical_to_cutedsl")
    if any(row[field] != "True" for row in operands
           for field in identical + (scales if row["precision"] == "nvfp4" else ())):
        problems.append("cuBLASLt operand bytes differ from the CuTe DSL operand bytes")
    plans = json.loads(files["raw/cublaslt_plans.json"].read_text(encoding="utf-8"))
    if len(plans) != len(configurations) or not all(
            plan.get("same_algorithm_for_every_set") and plan.get("identification_algorithm_matches")
            and plan.get("kernels_per_launch", 0) >= 1 and plan["algorithm"]["check_status"] == 0
            for plan in plans):
        problems.append("a cuBLASLt plan changed algorithm, lacks its kernel name or failed "
                        "the FP32-output check")
    cute = read_rows(files["precision_comparison.csv"])
    if len(cute) != len(configurations) or any(row["correctness"] != "PASS" for row in cute):
        problems.append("precision_comparison.csv is incomplete or not validated")
    for row in cute:
        summary = by_key.get((row["shape_id"], row["precision"], "cutedsl"), {})
        if any(row[f"repetition_{index}_tflops"] != summary.get(f"repetition_{index}_tflops")
               for index in range(1, REPETITIONS + 1)):
            problems.append(f"{row['shape_id']}/{row['precision']}: CuTe DSL repetitions differ "
                            "between the two CSVs")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gemm-profile", type=Path, required=True)
    parser.add_argument("--precision", type=Path, required=True)
    args = parser.parse_args()
    checks = {"gemm-profile": (check_gemm_profile, args.gemm_profile),
              "precision": (check_precision, args.precision)}
    failed = False
    for name, (check, directory) in checks.items():
        directory = directory if directory.is_absolute() else ROOT / directory
        problems = check(directory)
        failed |= bool(problems)
        print(f"{name}: {'FAIL' if problems else 'PASS'} {directory}")
        for problem in problems:
            print(f"  - {problem}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
