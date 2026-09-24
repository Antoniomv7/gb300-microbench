#!/usr/bin/env python3
"""Experiment V: BF16, FP8 and NVFP4 CuTe DSL GEMMs, compared with cuBLASLt on the same operands.

For every shape and format, three CuTe DSL repetitions run unchanged through the pinned CUTLASS
example's run(); cublaslt_precision.py then validates and times cuBLASLt on the operand bytes that
each repetition consumed. This script writes the raw records and refuses an incomplete or
mismatched run; analysis/analyze.py computes the two published summaries from the records.
"""

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import metadata  # noqa: E402

EXAMPLES = Path("/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel")
SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1),
          (32768, 512, 4096, 1))
FORMATS = ("bf16", "fp8", "nvfp4")
IMPLEMENTATIONS = ("cutedsl", "cublaslt")  # compared within each shape and format
REPETITIONS = 3
WARMUP_ITERATIONS = 5
ITERATIONS = 20
TILE = (256, 128)
CLUSTER = (2, 1)
REPETITION_FIELDS = ("shape_index", "shape_id", "m", "n", "k", "l", "precision",
                     "implementation", "repetition", "operand_set", "kernel_time_us", "tflops",
                     "validation", "bit_exact")
VALIDATION_FIELDS = ("shape_index", "shape_id", "m", "n", "k", "l", "precision", "operand_set",
                     "implementation", "stage", "reference", "status", "atol", "rtol", "finite",
                     "mismatches", "max_abs_error", "bit_exact",
                     "reference_values_not_bf16_exact", "reference_values_not_fp16_exact")
OPERAND_FIELDS = ("shape_index", "shape_id", "m", "n", "k", "l", "precision", "operand_set",
                  "operands_sha256", "a_bytes_identical_to_cutedsl", "b_bytes_identical_to_cutedsl",
                  "a_scale_bytes_identical_to_cutedsl", "b_scale_bytes_identical_to_cutedsl",
                  "represented_exactly", "outputs_bit_identical")


def load_example(name, relative):
    path = EXAMPLES / relative
    specification = importlib.util.spec_from_file_location(f"gb300_precision_{name}", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load pinned CuTe DSL example: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def measure(module, cutlass, precision, shape, warmup, iterations, verify):
    shared = {"mnkl": shape, "c_dtype": cutlass.Float32,
              "a_major": "k", "b_major": "k", "c_major": "n",
              "mma_tiler_mn": TILE, "cluster_shape_mn": CLUSTER,
              "use_tma_store": True, "warmup_iterations": warmup,
              "iterations": iterations, "skip_ref_check": not verify,
              "use_cold_l2": False}
    if precision == "nvfp4":
        # E4M3 scales over 16 values distinguish NVFP4 from MXFP4.
        shared.update(ab_dtype=cutlass.Float4E2M1FN,
                      sf_dtype=cutlass.Float8E4M3FN, sf_vec_size=16)
    else:
        shared.update(ab_dtype=cutlass.BFloat16 if precision == "bf16"
                      else cutlass.Float8E4M3FN,
                      acc_dtype=cutlass.Float32, use_2cta_instrs=True,
                      benchmark=True)
    microseconds = float(module.run(**shared))
    if not math.isfinite(microseconds) or microseconds <= 0:
        raise RuntimeError(f"{precision}/{shape}: invalid kernel time {microseconds}")
    return microseconds


def run(cublaslt):
    """Three CuTe DSL repetitions per shape and format, then cuBLASLt on their operand sets."""
    import cutlass
    import torch

    dense = load_example("dense", "dense_gemm/dense_gemm_persistent.py")
    blockscaled = load_example(
        "blockscaled", "blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py")
    for shape_index, shape in enumerate(SHAPES):
        for precision in FORMATS:
            torch.manual_seed(1111)
            module = blockscaled if precision == "nvfp4" else dense
            samples, captures = [], []
            for repetition in range(REPETITIONS):
                print(f"precision: {'x'.join(map(str, shape))}/{precision} "
                      f"repetition {repetition + 1}/{REPETITIONS}",
                      file=sys.stderr, flush=True)
                # The hooks only record the operands run() creates.
                with cublaslt.capture(module, precision) as captured:
                    # Validate each format and shape once before timing its repetitions.
                    samples.append(measure(module, cutlass, precision, shape, WARMUP_ITERATIONS,
                                           ITERATIONS, verify=repetition == 0))
                captures.append(captured)
            print(f"precision: {'x'.join(map(str, shape))}/{precision} cuBLASLt on the "
                  "same operand sets", file=sys.stderr, flush=True)
            cublaslt.measure(shape_index, shape, precision, cutlass, captures, samples)
            del captures


def write_table(path, fields, rows):
    # Full float precision lets analysis/analyze.py recompute this run's summaries exactly.
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_records(repetitions, validation, operands, plans):
    """Check the same evidence during acquisition and CPU-only reconstruction."""
    problems = []
    configurations = {(index, "x".join(map(str, shape)), name)
                      for index, shape in enumerate(SHAPES) for name in FORMATS}
    sets = ("validated", *(f"timed_{index}" for index in range(1, REPETITIONS + 1)))

    def require_keys(records, fields, expected, label):
        keys = [(int(row["shape_index"]), row["shape_id"], row["precision"],
                 *(row[field] for field in fields)) for row in records]
        if len(keys) != len(expected) or set(keys) != expected:
            problems.append(f"{label}: missing, duplicate or unexpected configuration records")

    require_keys(repetitions, ("implementation", "repetition", "operand_set"),
                 {(*key, method, index, f"timed_{index}") for key in configurations
                  for method in IMPLEMENTATIONS for index in range(1, REPETITIONS + 1)},
                 "repetitions")
    stages = {("cutedsl", "after_example_run", name) for name in sets} | \
             {("cublaslt", "before_timing", name) for name in sets} | \
             {("cublaslt", "after_timing", name) for name in sets[1:]}
    require_keys(validation, ("implementation", "stage", "operand_set"),
                 {(*key, *stage) for key in configurations for stage in stages}, "validation")
    require_keys(operands, ("operand_set",),
                 {(*key, name) for key in configurations for name in sets}, "operands")
    require_keys(plans, (), configurations, "cuBLASLt plans")
    for record in repetitions:
        if record["validation"] != "PASS" or any(
                not math.isfinite(record[field]) or record[field] <= 0
                for field in ("kernel_time_us", "tflops")):
            problems.append("a timed repetition failed validation or has an invalid measurement")
    for record in validation:
        if record["status"] != "PASS":
            problems.append(f"{record['shape_id']}/{record['precision']}/{record['operand_set']}: "
                            f"{record['implementation']} failed validation at {record['stage']}")
    for record in operands:
        # cuBLASLt must consume the very bytes, NVFP4 block scales included, that CuTe DSL did.
        names = ("a", "b", "a_scale", "b_scale") if record["precision"] == "nvfp4" else ("a", "b")
        if not (record["represented_exactly"] is True and
                all(record[f"{name}_bytes_identical_to_cutedsl"] is True for name in names)):
            problems.append(f"{record['shape_id']}/{record['precision']}/{record['operand_set']}: "
                            "cuBLASLt operands differ from the CuTe DSL operand bytes")
    for plan in plans:
        # One algorithm for every operand set, accepted by cublasLtMatmulAlgoCheck for FP32 output.
        if (plan["algorithm"]["check_status"] != 0 or
                plan["same_algorithm_for_every_set"] is not True or
                plan["identification_algorithm_matches"] is not True):
            problems.append(f"{plan['shape_id']}/{plan['precision']}: the cuBLASLt algorithm "
                            "changed or failed cublasLtMatmulAlgoCheck")
    if problems:
        raise ValueError("; ".join(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="new run directory; an existing directory is never reused")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    # A sibling of this script; analysis/analyze.py imports this file only for its constants.
    import cublaslt_precision

    record = {"experiment": "precision_comparison", "created_utc": metadata.utc_now(),
              **metadata.environment()}
    cublaslt = cublaslt_precision.Baseline(WARMUP_ITERATIONS, ITERATIONS)
    run(cublaslt)
    # Kernel names come from a profiler, so they are read only after every timed launch.
    cublaslt.identify_kernels()
    for plan in cublaslt.plans:
        if plan["kernel_identification_status"] == "UNAVAILABLE":
            print(f"precision: no CUDA kernel event captured for "
                  f"{plan['shape_id']}/{plan['precision']}; selected algorithm and "
                  "validation remain recorded", file=sys.stderr)

    raw = output / "raw"
    raw.mkdir()
    write_table(raw / "repetitions.csv", REPETITION_FIELDS, cublaslt.repetitions)
    write_table(raw / "validation.csv", VALIDATION_FIELDS, cublaslt.validation)
    write_table(raw / "operands.csv", OPERAND_FIELDS, cublaslt.operands)
    (raw / "cublaslt_plans.json").write_text(json.dumps(cublaslt.plans, indent=2) + "\n",
                                             encoding="utf-8")
    check_records(cublaslt.repetitions, cublaslt.validation, cublaslt.operands, cublaslt.plans)
    record.update({
        "parameters": {
            "shapes": ["x".join(map(str, shape)) for shape in SHAPES], "formats": list(FORMATS),
            "repetitions": REPETITIONS, "warmup_iterations": WARMUP_ITERATIONS,
            "iterations": ITERATIONS, "mma_tile": "x".join(map(str, TILE)),
            "cluster_shape": "x".join(map(str, CLUSTER)),
            "tolerances": {precision: {"atol": atol, "rtol": rtol} for precision, (atol, rtol)
                           in cublaslt_precision.TOLERANCES.items()},
            "cublaslt_workspace_limit_bytes": cublaslt_precision.WORKSPACE_LIMIT_BYTES,
            "cublaslt_requested_algorithms": cublaslt_precision.REQUESTED_ALGORITHMS},
        "cublaslt_version": cublaslt.bridge.version()})
    metadata.complete(record)
    (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"precision: complete {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ImportError, ValueError, AssertionError) as error:
        print(f"precision: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
