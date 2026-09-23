#!/usr/bin/env python3
"""Experiment V: matched BF16, FP8 and NVFP4 CuTe DSL GEMMs with a within-format cuBLASLt baseline.

One acquisition writes the nine-row CuTe DSL precision summary and the 18-row CuTe DSL/cuBLASLt
comparison, both from the same three repetitions of every format and shape.
"""

import argparse
import contextlib
import datetime as dt
import html
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import stats, svg_start, svg_text, write_csv
from scripts import provenance

EXAMPLES = Path("/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel")
SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1),
          (32768, 512, 4096, 1))
FORMATS = ("bf16", "fp8", "nvfp4")
COLORS = {"bf16": "#2563eb", "fp8": "#7c3aed", "nvfp4": "#d97706"}
LABELS = {"bf16": "BF16", "fp8": "FP8 E4M3", "nvfp4": "NVFP4 E2M1"}
VENDOR_DENSE_TFLOPS = {"bf16": 2250.0, "fp8": 4500.0, "nvfp4": 13500.0}
REPETITIONS = 3
WARMUP_ITERATIONS = 5
ITERATIONS = 20
TILE = (256, 128)
CLUSTER = (2, 1)
IMPLEMENTATIONS = ("cutedsl", "cublaslt")
# CuTe DSL keeps the persistent_2cta amber of the GEMM figure; the pair passes the CVD checks.
IMPLEMENTATION_COLORS = {"cutedsl": "#d97706", "cublaslt": "#166534"}
IMPLEMENTATION_LABELS = {"cutedsl": "CuTe DSL persistent 2-CTA",
                         "cublaslt": "cuBLASLt first supported heuristic"}
COMPARISON_FIELDS = (
    "shape_index", "shape_id", "m", "n", "k", "l", "precision", "implementation", "comparison",
    "input_dtype", "scale_dtype", "scale_block_elements", "scale_layout", "accumulator_dtype",
    "output_dtype", "alpha", "beta", "kernel", "repetition_1_tflops", "repetition_2_tflops",
    "repetition_3_tflops", "mean_tflops", "stdev_tflops", "cv_percent", "mean_kernel_time_us",
    "throughput_ratio_vs_cublaslt", "validation", "bit_exact_outputs")
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
SOURCES = ("precision_comparison/precision_comparison.py",
           "precision_comparison/cublaslt_precision.py",
           "precision_comparison/cublaslt_precision_bridge.cu", "analysis/analyze.py",
           "scripts/provenance.py")


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


def summarize(shape_index, shape, precision, samples):
    m, n, k, batch = shape
    throughputs = [2 * math.prod(shape) / duration / 1e6 for duration in samples]
    performance = stats(throughputs)
    peak = VENDOR_DENSE_TFLOPS[precision]
    return {"shape_index": shape_index, "shape_id": "x".join(map(str, shape)),
            "m": m, "n": n, "k": k, "l": batch, "precision": precision,
            "input_dtype": "Float4E2M1FN" if precision == "nvfp4" else
            "Float8E4M3FN" if precision == "fp8" else "BFloat16",
            "scale_dtype": "Float8E4M3FN" if precision == "nvfp4" else "",
            "scale_block_elements": 16 if precision == "nvfp4" else "",
            "accumulator_dtype": "Float32", "output_dtype": "Float32",
            "mma_tile": "256x128", "cluster_shape": "2x1", "tma_store": "yes",
            **{f"repetition_{index}_tflops": value
               for index, value in enumerate(throughputs, 1)},
            "mean_tflops": performance["mean"],
            "stdev_tflops": performance["stdev_sample"],
            "cv_percent": performance["cv_percent"],
            "mean_kernel_time_us": stats(samples)["mean"],
            "speedup_vs_bf16": "", "vendor_dense_peak_tflops": peak,
            "percent_vendor_dense_peak": 100 * performance["mean"] / peak,
            "correctness": "PASS"}


def figure(rows):
    width, height, left, right, top, bottom = 1260, 500, 86, 38, 140, 395
    output = svg_start("Low-precision GEMM: BF16 versus FP8 versus NVFP4",
                       "Three repeated hot-cache measurements per matched configuration.",
                       width, height)
    shapes = sorted({(row["shape_index"], row["shape_id"]) for row in rows})
    maximum = max(row[f"repetition_{index}_tflops"]
                  for row in rows for index in range(1, REPETITIONS + 1)) * 1.12
    y = lambda value: bottom - value * (bottom - top) / maximum

    for tick in range(6):
        value = maximum * tick / 5
        yy = y(value)
        output.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(left - 9, yy + 4, f"{value:,.0f}",
                               text_anchor="end", font_size="10", fill="#64748b"))

    for index, precision in enumerate(FORMATS):
        xx = 685 + index * 165
        output.append(f'<rect x="{xx}" y="83" width="12" height="12" '
                      f'fill="{COLORS[precision]}"/>')
        output.append(svg_text(xx + 18, 94, LABELS[precision], font_size="11"))

    lookup = {(row["shape_index"], row["precision"]): row for row in rows}
    group_width = (width - left - right) / len(shapes)
    bar_width = (group_width - 70) / len(FORMATS)
    for index, (shape_index, shape_id) in enumerate(shapes):
        for position, precision in enumerate(FORMATS):
            row = lookup[(shape_index, precision)]
            samples = [row[f"repetition_{sample}_tflops"]
                       for sample in range(1, REPETITIONS + 1)]
            xx = left + index * group_width + 30 + position * (bar_width + 3)
            yy = y(row["mean_tflops"])
            output.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" '
                          f'height="{bottom-yy:.1f}" fill="{COLORS[precision]}"/>')
            center = xx + bar_width / 2
            output.append(f'<line x1="{center:.1f}" y1="{y(min(samples)):.1f}" '
                          f'x2="{center:.1f}" y2="{y(max(samples)):.1f}" '
                          'stroke="#0f172a"/>')
        label = "×".join(shape_id.removesuffix("x1").split("x"))
        output.append(svg_text(left + (index + 0.5) * group_width, bottom + 26,
                               label, text_anchor="middle", font_size="11"))

    output.append(svg_text(19, 268, "TFLOP/s", text_anchor="middle",
                           transform="rotate(-90 19 268)"))
    output.append(svg_text(34, 470,
                           "FP32 accumulation/output; tile 256×128; cluster 2×1; "
                           "NVFP4 uses one E4M3 scale per 16 values.",
                           font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def tick_step(span, count=5):
    """Smallest round step (1 to 5 times a power of ten) giving at most count intervals."""
    magnitude = 10 ** math.floor(math.log10(span / count))
    return next(step * magnitude for step in (1, 1.5, 2, 2.5, 3, 4, 5, 10)
                if step * magnitude * count >= span)


def comparison_figure(rows, warmup, iterations):
    width, height, left, right, top, bottom = 1260, 540, 86, 38, 140, 408
    output = svg_start("CuTe DSL versus cuBLASLt by precision",
                       "Same logical operands and FP32 output within each format; mean of three "
                       "repetitions, whiskers show their range.", width, height)
    shapes = sorted({(row["shape_index"], row["shape_id"]) for row in rows})
    lookup = {(row["shape_index"], row["precision"], row["implementation"]): row for row in rows}
    samples = lambda row: [row[f"repetition_{index}_tflops"]
                           for index in range(1, REPETITIONS + 1)]
    highest = max(max(samples(row)) for row in rows) * 1.10
    step = tick_step(highest)
    maximum = step * math.ceil(highest / step)
    y = lambda value: bottom - value * (bottom - top) / maximum

    for tick in range(round(maximum / step) + 1):
        value = step * tick
        yy = y(value)
        output.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                      f'y2="{yy:.1f}" stroke="#e2e8f0"/>')
        output.append(svg_text(left - 9, yy + 4, f"{value:,.0f}",
                               text_anchor="end", font_size="10", fill="#64748b"))
    for index, implementation in enumerate(IMPLEMENTATIONS):
        xx = 640 + index * 300
        output.append(f'<rect x="{xx}" y="83" width="12" height="12" '
                      f'fill="{IMPLEMENTATION_COLORS[implementation]}"/>')
        output.append(svg_text(xx + 18, 94, IMPLEMENTATION_LABELS[implementation],
                               font_size="11"))

    group_width = (width - left - right) / len(shapes)
    bar_width, bar_gap, pair_gap = 24, 2, 64
    pair_width = 2 * bar_width + bar_gap
    span = len(FORMATS) * pair_width + (len(FORMATS) - 1) * pair_gap
    for index, (shape_index, shape_id) in enumerate(shapes):
        start = left + index * group_width + (group_width - span) / 2
        for position, precision in enumerate(FORMATS):
            x0 = start + position * (pair_width + pair_gap)
            pair = [lookup[(shape_index, precision, implementation)]
                    for implementation in IMPLEMENTATIONS]
            for offset, row in enumerate(pair):
                xx = x0 + offset * (bar_width + bar_gap)
                yy = y(row["mean_tflops"])
                values = samples(row)
                title = (f"{IMPLEMENTATION_LABELS[row['implementation']]}, {LABELS[precision]}, "
                         f"{shape_id}: {row['mean_tflops']:,.1f} TFLOP/s "
                         f"(range {min(values):,.1f} to {max(values):,.1f})")
                output.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width}" '
                              f'height="{bottom-yy:.1f}" '
                              f'fill="{IMPLEMENTATION_COLORS[row["implementation"]]}">'
                              f'<title>{html.escape(title)}</title></rect>')
                center = xx + bar_width / 2
                output.append(f'<line x1="{center:.1f}" y1="{y(min(values)):.1f}" '
                              f'x2="{center:.1f}" y2="{y(max(values)):.1f}" stroke="#0f172a"/>')
            highest = max(max(samples(row)) for row in pair)
            output.append(svg_text(x0 + pair_width / 2, y(highest) - 8,
                                   f"{pair[0]['throughput_ratio_vs_cublaslt']:.2f}×",
                                   text_anchor="middle", font_size="11", fill="#334155"))
            output.append(svg_text(x0 + pair_width / 2, bottom + 18, LABELS[precision],
                                   text_anchor="middle", font_size="11"))
        label = "×".join(shape_id.removesuffix("x1").split("x"))
        output.append(svg_text(left + (index + 0.5) * group_width, bottom + 42,
                               label, text_anchor="middle", font_size="12", font_weight="700"))

    output.append(svg_text(19, 274, "TFLOP/s", text_anchor="middle",
                           transform="rotate(-90 19 274)"))
    output.append(svg_text(34, 508,
                           "Ratio above each pair: CuTe DSL / cuBLASLt mean throughput. Each "
                           f"repetition times {iterations} hot-cache launches after {warmup} "
                           "warm-up launches.", font_size="11", fill="#64748b"))
    return "\n".join([*output, "</svg>"]) + "\n"


def run(shapes, warmup, iterations, cublaslt):
    import cutlass
    import torch

    dense = load_example("dense", "dense_gemm/dense_gemm_persistent.py")
    blockscaled = load_example(
        "blockscaled", "blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py")
    rows = []
    for shape_index, shape in enumerate(shapes):
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
                    samples.append(measure(module, cutlass, precision, shape,
                                           warmup, iterations, verify=repetition == 0))
                captures.append(captured)
            rows.append(summarize(shape_index, shape, precision, samples))
            print(f"precision: {'x'.join(map(str, shape))}/{precision} cuBLASLt on the "
                  "same operand sets", file=sys.stderr, flush=True)
            cublaslt.measure(shape_index, shape, precision, cutlass, captures, samples)
            del captures

        baseline = next(row for row in rows
                        if row["shape_index"] == shape_index and row["precision"] == "bf16")
        for row in rows:
            if row["shape_index"] == shape_index:
                row["speedup_vs_bf16"] = row["mean_tflops"] / baseline["mean_tflops"]
    return rows


def write_table(path, fields, rows):
    """Write rows in a fixed column order, formatting floats like the published CSVs."""
    write_csv(path, [{field: row.get(field, "") for field in fields} for row in rows])


def comparison_rows(cublaslt):
    """One row per shape, format and implementation of the within-format comparison."""
    from cublaslt_precision import CUTE_KERNELS, arithmetic

    plans = {(entry["shape_index"], entry["precision"]): entry for entry in cublaslt.plans}
    keys = sorted({(record["shape_index"], record["precision"])
                   for record in cublaslt.repetitions},
                  key=lambda key: (key[0], FORMATS.index(key[1])))
    rows = []
    for shape_index, precision in keys:
        same = lambda record, implementation: (
            (record["shape_index"], record["precision"], record["implementation"]) ==
            (shape_index, precision, implementation))
        summaries = {}
        for implementation in IMPLEMENTATIONS:
            samples = sorted((record for record in cublaslt.repetitions
                              if same(record, implementation)),
                             key=lambda record: record["repetition"])
            checks = [record for record in cublaslt.validation if same(record, implementation)]
            throughputs = [record["tflops"] for record in samples]
            summaries[implementation] = {
                "samples": samples, "throughput": stats(throughputs),
                "time": stats([record["kernel_time_us"] for record in samples]),
                "validation": "PASS" if len(samples) == REPETITIONS and checks and all(
                    check["status"] == "PASS" for check in checks) else "FAIL",
                "bit_exact": all(check["bit_exact"] for check in checks)}
        plan = plans.get((shape_index, precision), {})
        # Matched: both validated against the same reference, FP32 output confirmed by
        # cublasLtMatmulAlgoCheck, and one algorithm across every operand set.
        matched = (all(summary["validation"] == "PASS" for summary in summaries.values()) and
                   plan.get("algorithm", {}).get("check_status") == 0 and
                   plan.get("same_algorithm_for_every_set", False) and
                   plan.get("identification_algorithm_matches", False))
        cublaslt_mean = summaries["cublaslt"]["throughput"]["mean"]
        for implementation in IMPLEMENTATIONS:
            summary = summaries[implementation]
            first = summary["samples"][0]
            if implementation == "cutedsl":
                kernel = (f"{CUTE_KERNELS[precision]}; MMA tile {TILE[0]}x{TILE[1]}; "
                          f"cluster {CLUSTER[0]}x{CLUSTER[1]}; TMA store")
            else:
                algorithm = plan["algorithm"]
                name = " + ".join(plan.get("kernel_names", [])) or "name unavailable in profiler"
                kernel = (f"{name} (algorithm "
                          f"{algorithm['algorithm_id']}; tile {algorithm['tile_id']}; stages "
                          f"{algorithm['stages_id']}; cluster {algorithm['cluster_shape_id']})")
            rows.append({
                **{key: first[key] for key in ("shape_index", "shape_id", "m", "n", "k", "l",
                                               "precision")},
                "implementation": implementation,
                "comparison": "matched" if matched else "not_matched",
                **arithmetic(implementation, precision), "kernel": kernel,
                **{f"repetition_{record['repetition']}_tflops": record["tflops"]
                   for record in summary["samples"]},
                "mean_tflops": summary["throughput"]["mean"],
                "stdev_tflops": summary["throughput"]["stdev_sample"],
                "cv_percent": summary["throughput"]["cv_percent"],
                "mean_kernel_time_us": summary["time"]["mean"],
                "throughput_ratio_vs_cublaslt": summary["throughput"]["mean"] / cublaslt_mean,
                "validation": summary["validation"], "bit_exact_outputs": summary["bit_exact"]})
    return rows


def write_extended(output, shapes, rows, cublaslt, environment, created, warmup, iterations):
    """Write both summaries, the raw records and the metadata; decide whether it is complete."""
    from cublaslt_precision import (ARITHMETIC, CUTE_KERNELS, REFERENCE, REQUESTED_ALGORITHMS,
                                    TOLERANCES, WORKSPACE_LIMIT_BYTES, arithmetic)

    raw = output / "raw"
    raw.mkdir()
    write_table(raw / "repetitions.csv", REPETITION_FIELDS, cublaslt.repetitions)
    write_table(raw / "validation.csv", VALIDATION_FIELDS, cublaslt.validation)
    write_table(raw / "operands.csv", OPERAND_FIELDS, cublaslt.operands)
    (raw / "cublaslt_plans.json").write_text(json.dumps(cublaslt.plans, indent=2) + "\n",
                                             encoding="utf-8")
    # The CuTe DSL summary keeps the published schema of results/precision_comparison.csv.
    write_csv(output / "precision_comparison.csv", rows)
    (output / "precision_comparison.svg").write_text(figure(rows), encoding="utf-8")
    comparison = comparison_rows(cublaslt) if not cublaslt.limitations else []
    if comparison:
        write_table(output / "precision_cutedsl_vs_cublaslt.csv", COMPARISON_FIELDS, comparison)
        (output / "precision_cutedsl_vs_cublaslt.svg").write_text(
            comparison_figure(comparison, warmup, iterations), encoding="utf-8")

    configurations = len(shapes) * len(FORMATS)
    failures = [record for record in cublaslt.validation if record["status"] != "PASS"]
    problems = [f"cuBLASLt limitation: {item['shape_id']}/{item['precision']}: {item['reason']}"
                for item in cublaslt.limitations]
    if len(cublaslt.repetitions) != 2 * REPETITIONS * configurations:
        problems.append(f"{len(cublaslt.repetitions)} repetition rows, expected "
                        f"{2 * REPETITIONS * configurations}")
    if failures:
        problems.append(f"{len(failures)} validation records failed")
    if len(rows) != configurations:
        problems.append(f"{len(rows)} CuTe DSL summary rows, expected {configurations}")
    if len(comparison) != 2 * configurations:
        problems.append(f"{len(comparison)} comparison rows, expected {2 * configurations}")
    problems += [f"{row['shape_id']}/{row['precision']}: comparison not matched"
                 for row in comparison if row["comparison"] != "matched"
                 and row["implementation"] == "cublaslt"]
    scaled = [record for record in cublaslt.operands if record["precision"] == "nvfp4"]
    metadata = {
        "experiment": "precision_comparison: BF16, FP8 and NVFP4 CuTe DSL GEMMs with a "
                      "within-format cuBLASLt baseline",
        "state": "COMPLETE" if not problems else "INCOMPLETE", "problems": problems,
        "created_utc": created, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "environment": environment,
        "protocol": {
            "shapes": ["x".join(map(str, shape)) for shape in shapes], "formats": list(FORMATS),
            "repetitions": REPETITIONS, "warmup_iterations": warmup, "iterations": iterations,
            "timing": "cute.testing.benchmark: warm-up launches, then CUDA events on the default "
                      "stream around the timed launches; the result is the mean launch time",
            "cache_state": "hot L2 (one workspace; use_cold_l2=False)",
            "order": "per shape and format: three CuTe DSL repetitions through the pinned "
                     "example's run(), then cuBLASLt on the validated set and on each "
                     "repetition's timed set",
            "outside_timed_interval": ["operand generation", "conversion and packing",
                                       "scale layout transformation",
                                       "plan selection or compilation", "reference", "validation"]},
        "operands": "cuBLASLt consumes the logical operands that each CuTe DSL repetition drew, "
                    "re-encoded from those values; the NVFP4 repetitions reseed inside run() and "
                    "therefore share operand values, the BF16 and FP8 repetitions do not",
        "reference": {"definition": REFERENCE,
                      "tolerances": {precision: {"atol": atol, "rtol": rtol}
                                     for precision, (atol, rtol) in TOLERANCES.items()}},
        "cutedsl": {"mma_tile": "x".join(map(str, TILE)),
                    "cluster_shape": "x".join(map(str, CLUSTER)),
                    "tma_store": True, "kernels": CUTE_KERNELS,
                    "validation": "the pinned example's own check of repetition 1's first launch, "
                                  "then every output it left behind against the reference"},
        "cublaslt": {
            "version": cublaslt.bridge.version(),
            "libraries": provenance.loaded_library("libcublasLt"),
            "selection_policy": f"first result with state CUBLAS_STATUS_SUCCESS among up to "
                                f"{REQUESTED_ALGORITHMS} from cublasLtMatmulAlgoGetHeuristic "
                                "(CUBLASLT_SEARCH_BEST_FIT); no timed search; one plan per "
                                "operand set",
            "workspace_limit_bytes": WORKSPACE_LIMIT_BYTES,
            "formulation": "column-major TN problem D^T = B A^T on the row-major, K-major "
                           "operands; FP32 C/D with beta = 0",
            "fp32_output_check": "cublasLtMatmulAlgoCheck on every selected algorithm with "
                                 "CUDA_R_32F C and D",
            "fast_accumulation": 0, "pointer_mode": "host",
            "kernel_identification": "one CPU + CUDA torch.profiler capture after timing; "
                                     "UNAVAILABLE means the trace contained no kernel "
                                     "event, not that no kernel executed",
            "kernel_names_unavailable": [f"{entry['shape_id']}/{entry['precision']}"
                                         for entry in cublaslt.plans if
                                         entry["kernel_identification_status"] == "UNAVAILABLE"],
            "epilogue": "CUBLASLT_EPILOGUE_DEFAULT",
            "selected": [{key: entry.get(key) for key in (
                "shape_id", "precision", "algorithm", "same_algorithm_for_every_set",
                "identification_algorithm_matches", "kernel_identification_status",
                "kernel_names", "kernels_per_launch")}
                for entry in cublaslt.plans]},
        "arithmetic": {implementation: {precision: arithmetic(implementation, precision)
                                        for precision in FORMATS}
                       for implementation in ARITHMETIC},
        "checks": {
            "repetition_rows": len(cublaslt.repetitions),
            "validation_records": len(cublaslt.validation), "validation_failures": len(failures),
            "operand_sets": len(cublaslt.operands),
            "data_bytes_identical_to_cutedsl": all(
                record["a_bytes_identical_to_cutedsl"] and record["b_bytes_identical_to_cutedsl"]
                for record in cublaslt.operands),
            "nvfp4_scale_bytes_identical_to_cutedsl": all(
                record["a_scale_bytes_identical_to_cutedsl"] and
                record["b_scale_bytes_identical_to_cutedsl"] for record in scaled),
            "represented_exactly": all(record["represented_exactly"]
                                       for record in cublaslt.operands),
            "outputs_bit_identical": all(record.get("outputs_bit_identical", False)
                                         for record in cublaslt.operands)},
        "limitations": cublaslt.limitations,
        "files": sorted(str(path.relative_to(output)) for path in output.rglob("*")
                        if path.is_file()) + ["metadata.json"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
    return metadata["state"]


@contextlib.contextmanager
def logged(path):
    """Mirror everything this process prints into the run log."""
    streams = sys.stdout, sys.stderr

    class Tee:
        def __init__(self, stream, log):
            self.stream, self.log = stream, log

        def write(self, text):
            self.log.write(text)
            return self.stream.write(text)

        def flush(self):
            self.log.flush()
            self.stream.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    with path.open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(streams[0], log), Tee(streams[1], log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = streams


def run_extended(output, shapes, warmup, iterations):
    import cublaslt_precision

    created = dt.datetime.now(dt.timezone.utc).isoformat()
    environment = {"gpu": provenance.gpu_identity(), "software": provenance.software_versions(),
                   "repository": provenance.repository_state(SOURCES),
                   "pinned": provenance.pinned_versions()}
    cublaslt = cublaslt_precision.Baseline(warmup, iterations)
    rows = run(shapes, warmup, iterations, cublaslt)
    # Kernel names come from a profiler, so they are read only after every timed launch.
    cublaslt.identify_kernels()
    for entry in cublaslt.plans:
        if entry["kernel_identification_status"] == "UNAVAILABLE":
            print(f"precision: no CUDA kernel event captured for "
                  f"{entry['shape_id']}/{entry['precision']}; selected algorithm and "
                  "validation remain recorded", file=sys.stderr)
    return write_extended(output, shapes, rows, cublaslt, environment, created,
                          warmup, iterations)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="new run directory; an existing directory is never reused")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    with logged(output / "run.log"):
        state = run_extended(output, SHAPES, WARMUP_ITERATIONS, ITERATIONS)
        print(f"precision: {state} {output}", file=sys.stderr)
    if state != "COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ImportError, ValueError, AssertionError) as error:
        print(f"precision: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
