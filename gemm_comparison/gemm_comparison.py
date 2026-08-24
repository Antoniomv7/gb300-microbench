#!/usr/bin/env python3
"""Compare three BF16 CuTe DSL kernels with the cuBLASLt heuristic."""

import argparse
import csv
import ctypes
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1),
          (16384, 512, 4096, 1), (32768, 512, 4096, 1),
          (512, 16384, 4096, 1))
CANDIDATES = (
    {"method": "cutedsl", "variant": "nonpersistent_1cta", "module": "nonpersistent",
     "class": "DenseGemmKernel", "tile": (128, 128), "cluster": (1, 1),
     "persistent": False, "use_2cta": False},
    {"method": "cutedsl", "variant": "persistent_1cta", "module": "persistent",
     "class": "PersistentDenseGemmKernel", "tile": (128, 128), "cluster": (1, 1),
     "persistent": True, "use_2cta": False},
    {"method": "cutedsl", "variant": "persistent_2cta", "module": "persistent",
     "class": "PersistentDenseGemmKernel", "tile": (256, 128), "cluster": (2, 1),
     "persistent": True, "use_2cta": True},
    {"method": "cublaslt", "variant": "heuristic_first_supported"},
)
FIELDS = ("shape_index", "shape_id", "m", "n", "k", "l", "candidate_index",
          "method", "variant", "correctness", "mismatches", "max_abs_error",
          "max_rel_error", "warmup_iterations", "iterations", "compile_time_ms",
          "setup_time_ms", "first_launch_ms", "kernel_time_ms", "tflops",
          "throughput_ratio_vs_cublaslt", "gap_to_cublaslt_pct",
          "best_cutedsl_variant", "rank_within_shape", "workspace_bytes",
          "heuristic_index", "algorithm_id")
EXAMPLES = Path("/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm")
BRIDGE_LIBRARY = Path("/tmp/gb300-cublaslt/libcublaslt_bridge.so")
ATOL, RTOL = 0.1, 1e-5


class PlanInfo(ctypes.Structure):
    _fields_ = [("workspace_bytes", ctypes.c_int64),
                ("heuristic_index", ctypes.c_int64),
                ("algorithm_id", ctypes.c_int64)]


class CublasLtBridge:
    def __init__(self):
        self.library = ctypes.CDLL(str(BRIDGE_LIBRARY))
        self.library.gb_last_error.restype = ctypes.c_char_p
        self.library.gb_plan_create.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 5 + [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(PlanInfo)]
        self.library.gb_plan_execute.argtypes = [ctypes.c_void_p]
        self.library.gb_plan_destroy.argtypes = [ctypes.c_void_p]
        self.plan = ctypes.c_void_p()

    def check(self, status):
        if status:
            raise RuntimeError(self.library.gb_last_error().decode())

    def create(self, m, n, k, a, b, c, stream):
        info = PlanInfo()
        self.check(self.library.gb_plan_create(
            m, n, k, a.data_ptr(), b.data_ptr(), c.data_ptr(), c.data_ptr(),
            stream.cuda_stream, ctypes.byref(self.plan), ctypes.byref(info)))
        return info

    def execute(self):
        self.check(self.library.gb_plan_execute(self.plan))

    def close(self):
        if self.plan.value:
            self.check(self.library.gb_plan_destroy(self.plan))
            self.plan = ctypes.c_void_p()


def load_module(name, filename):
    path = EXAMPLES / filename
    specification = importlib.util.spec_from_file_location(f"gb300_{name}", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load CuTe DSL example: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def check_layout(tensor, name, shape, strides, dtype):
    if (tensor.device.type != "cuda" or tensor.dtype != dtype or
            tuple(tensor.shape) != shape or tuple(tensor.stride()) != strides):
        raise RuntimeError(f"{name}: incompatible dtype, shape or row-major strides")


def create_operands(torch, cutlass, factory, shape):
    m, n, k, l = shape
    torch.manual_seed(1111)
    a, b, c, a_host, b_host, _, output = factory.create_tensors(
        l, m, n, k, "k", "k", "n", cutlass.BFloat16, cutlass.Float32)
    a_gpu = torch.empty_like(a_host, device="cuda")
    b_gpu = torch.empty_like(b_host, device="cuda")
    a_gpu.copy_(a_host)
    b_gpu.copy_(b_host)
    check_layout(a_gpu, "A", (m, k, l), (k, 1, m * k), torch.bfloat16)
    check_layout(b_gpu, "B", (n, k, l), (k, 1, n * k), torch.bfloat16)
    check_layout(output, "C", (m, n, l), (n, 1, m * n), torch.float32)
    return {"a": a, "b": b, "c": c, "a_host": a_host, "b_host": b_host,
            "a_gpu": a_gpu, "b_gpu": b_gpu, "output": output}


def reference_result(torch, operands):
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "ieee"
        result = torch.einsum(
            "mkl,nkl->mnl",
            operands["a_host"].to(device="cuda", dtype=torch.float32),
            operands["b_host"].to(device="cuda", dtype=torch.float32))
    finally:
        matmul.fp32_precision = previous
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("the FP32 reference contains non-finite values")
    return result


def validate_result(torch, output, reference, label):
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"{label}: incomplete or non-finite output")
    difference = (output - reference).abs()
    mismatches = int((difference > ATOL + RTOL * reference.abs()).sum().item())
    absolute = float(difference.max().item())
    relative = float((difference / reference.abs().clamp_min(1.0)).max().item())
    if mismatches or not math.isfinite(absolute) or not math.isfinite(relative):
        raise RuntimeError(f"{label}: {mismatches} mismatches; maximum error {absolute}")
    return absolute, relative


def prepare_cutedsl(specification, modules, shape, operands, cute, cutlass, stream):
    kernel = getattr(modules[specification["module"]], specification["class"])(
        cutlass.Float32, specification["use_2cta"], specification["tile"],
        specification["cluster"], True)
    tensors = (operands["a"], operands["b"], operands["c"])
    if specification["persistent"]:
        import cutlass.utils as utils

        supported = kernel.can_implement(shape, *(tensor.element_type for tensor in tensors),
                                         "k", "k", "n")
        cluster_size = math.prod(specification["cluster"])
        active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        compile_args = (*tensors, active_clusters, stream)
    else:
        supported = kernel.can_implement(*tensors)
        compile_args = (*tensors, stream)
    if not supported:
        raise RuntimeError(f"{specification['variant']}: unsupported configuration")
    started = time.perf_counter_ns()
    compiled = cute.compile(kernel, *compile_args)
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    return lambda: compiled(*tensors, stream), {"compile_time_ms": compile_ms}


def measure_candidate(specification, modules, shape, operands, reference, context, warmup, count):
    torch, cutlass, cute, torch_stream, cute_stream = context
    label = f"{'x'.join(map(str, shape))}/{specification['variant']}"
    print(f"gemm: {label}", file=sys.stderr, flush=True)
    operands["output"].fill_(float("nan"))
    torch.cuda.synchronize()
    bridge = None
    try:
        if specification["method"] == "cutedsl":
            launch, details = prepare_cutedsl(specification, modules, shape, operands,
                                              cute, cutlass, cute_stream)
        else:
            bridge = CublasLtBridge()
            started = time.perf_counter_ns()
            info = bridge.create(*shape[:3], operands["a_gpu"], operands["b_gpu"],
                                 operands["output"], torch_stream)
            details = {"setup_time_ms": (time.perf_counter_ns() - started) / 1e6,
                       "workspace_bytes": info.workspace_bytes,
                       "heuristic_index": info.heuristic_index,
                       "algorithm_id": info.algorithm_id}
            launch = bridge.execute

        started = time.perf_counter_ns()
        launch()
        torch.cuda.synchronize()
        details["first_launch_ms"] = (time.perf_counter_ns() - started) / 1e6
        absolute, relative = validate_result(torch, operands["output"], reference, label)
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(torch_stream)
        for _ in range(count):
            launch()
        end.record(torch_stream)
        torch.cuda.synchronize()
        duration = start.elapsed_time(end) / count
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"{label}: invalid CUDA-event duration")
        return {**details, "correctness": "PASS", "mismatches": 0,
                "max_abs_error": absolute, "max_rel_error": relative,
                "kernel_time_ms": duration,
                "tflops": 2 * math.prod(shape) / duration / 1e9}
    finally:
        if bridge is not None:
            bridge.close()


def run(warmup, iterations):
    import cutlass
    import cutlass.cute as cute
    import torch
    from cuda.bindings import driver

    modules = {"nonpersistent": load_module("dense_gemm", "dense_gemm.py"),
               "persistent": load_module("dense_gemm_persistent", "dense_gemm_persistent.py")}
    torch_stream = torch.cuda.current_stream()
    context = (torch, cutlass, cute, torch_stream, driver.CUstream(torch_stream.cuda_stream))
    rows = []
    for shape_index, shape in enumerate(SHAPES):
        operands = create_operands(torch, cutlass, modules["nonpersistent"], shape)
        reference = reference_result(torch, operands)
        measurements = [measure_candidate(candidate, modules, shape, operands, reference,
                                          context, warmup, iterations)
                        for candidate in CANDIDATES]
        baseline = measurements[-1]["tflops"]
        best = max(range(3), key=lambda index: measurements[index]["tflops"])
        order = sorted(range(4), key=lambda index: measurements[index]["tflops"], reverse=True)
        for index, (candidate, measured) in enumerate(zip(CANDIDATES, measurements)):
            ratio = measured["tflops"] / baseline
            rows.append({"shape_index": shape_index, "shape_id": "x".join(map(str, shape)),
                         **dict(zip(("m", "n", "k", "l"), shape)),
                         "candidate_index": index, "method": candidate["method"],
                         "variant": candidate["variant"], "warmup_iterations": warmup,
                         "iterations": iterations, **measured,
                         "throughput_ratio_vs_cublaslt": ratio,
                         "gap_to_cublaslt_pct": 100 * (1 - ratio),
                         "best_cutedsl_variant": CANDIDATES[best]["variant"],
                         "rank_within_shape": order.index(index) + 1})
        del reference, operands, measurements
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare BF16 CuTe DSL GEMM with cuBLASLt.")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_iterations < 0 or args.iterations <= 0:
        parser.error("warm-up must be non-negative and iterations must be positive")

    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        rows = run(args.warmup_iterations, args.iterations)
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ImportError) as error:
        print(f"gemm: ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
