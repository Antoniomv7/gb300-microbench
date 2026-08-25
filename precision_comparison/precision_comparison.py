#!/usr/bin/env python3
"""Compare matched BF16, FP8 and block-scaled NVFP4 CuTe DSL GEMMs."""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.analyze import stats, svg_start, svg_text, write_csv

EXAMPLES = Path("/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel")
SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1),
          (32768, 512, 4096, 1))
FORMATS = ("bf16", "fp8", "nvfp4")
COLORS = {"bf16": "#2563eb", "fp8": "#7c3aed", "nvfp4": "#d97706"}
LABELS = {"bf16": "BF16", "fp8": "FP8 E4M3", "nvfp4": "NVFP4 E2M1"}
VENDOR_DENSE_TFLOPS = {"bf16": 2250.0, "fp8": 4500.0, "nvfp4": 13500.0}
REPETITIONS = 3
TILE = (256, 128)
CLUSTER = (2, 1)


def parse_shape(value):
    try:
        dimensions = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must contain three integers") from error
    if len(dimensions) != 3 or any(dimension <= 0 or dimension % 32 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape dimensions must be positive multiples of 32")
    return (*dimensions, 1)


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


def run(shapes, warmup, iterations):
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
            samples = []
            for repetition in range(REPETITIONS):
                print(f"precision: {'x'.join(map(str, shape))}/{precision} "
                      f"repetition {repetition + 1}/{REPETITIONS}",
                      file=sys.stderr, flush=True)
                # Validate each format and shape once before timing its repetitions.
                samples.append(measure(module, cutlass, precision, shape,
                                       warmup, iterations, verify=repetition == 0))
            rows.append(summarize(shape_index, shape, precision, samples))

        baseline = next(row for row in rows
                        if row["shape_index"] == shape_index and row["precision"] == "bf16")
        for row in rows:
            if row["shape_index"] == shape_index:
                row["speedup_vs_bf16"] = row["mean_tflops"] / baseline["mean_tflops"]
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matched BF16, FP8 and NVFP4 persistent GEMMs.")
    parser.add_argument("--shape", action="append", type=parse_shape,
                        help="M,N,K; may be repeated; defaults to three thesis shapes")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.warmup_iterations < 0 or args.iterations <= 0:
        parser.error("warm-up must be non-negative and iterations must be positive")
    shapes = tuple(args.shape or SHAPES)
    if len(set(shapes)) != len(shapes):
        parser.error("shapes must be distinct")

    rows = run(shapes, args.warmup_iterations, args.iterations)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "precision_comparison.csv", rows)
    (output / "precision_comparison.svg").write_text(figure(rows), encoding="utf-8")
    print(f"precision: COMPLETE {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ImportError, ValueError, AssertionError) as error:
        print(f"precision: ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
