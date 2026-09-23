#!/usr/bin/env python3
"""cuBLASLt baselines measured on the CuTe DSL precision runs' own logical operands.

Each CuTe DSL repetition still runs unchanged through the pinned example's run(). Observation
hooks record the logical operands that run() draws, the bytes it hands to the kernel and the
kernel's output. The cuBLASLt baseline re-encodes the same logical operands in the formats and
scale layout that cuBLASLt documents, validates against a reference formed from its own decoded
bytes, and is timed by the same cute.testing.benchmark call as the CuTe DSL kernels.
"""

import contextlib
import ctypes
import hashlib
import inspect
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_LIBRARY = ROOT / "build/precision_comparison/libcublaslt_precision_bridge.so"
WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024  # the campaign cuBLASLt baseline's limit
REQUESTED_ALGORITHMS = 32
FORMAT_CODES = {"bf16": 0, "fp8": 1, "nvfp4": 2}
OUTPUT_FLOAT32 = 0
SCALE_BLOCK = 16
# Tolerances of the pinned examples' own checks (dense_gemm_persistent and the SM103 NVFP4 kernel).
TOLERANCES = {"bf16": (0.1, 1e-3), "fp8": (0.1, 1e-3), "nvfp4": (0.1, 1e-2)}
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REFERENCE = "IEEE-FP32 product of the dequantized, exactly represented operands"
CUTE_KERNELS = {"bf16": "PersistentDenseGemmKernel (dense_gemm_persistent.py)",
                "fp8": "PersistentDenseGemmKernel (dense_gemm_persistent.py)",
                "nvfp4": "Sm103BlockScaledPersistentDenseGemmKernel "
                         "(sm103_dense_blockscaled_gemm_persistent.py)"}
ARITHMETIC = {
    "cutedsl": {
        "input_dtype": {"bf16": "BFloat16", "fp8": "Float8E4M3FN", "nvfp4": "Float4E2M1FN"},
        "scale_dtype": {"nvfp4": "Float8E4M3FN"},
        "scale_layout": {"nvfp4": "32x4x4 atoms; 128 rows x 4 blocks per 512 B; K blocks fastest"},
        "accumulator_dtype": "Float32", "output_dtype": "Float32",
        "alpha": "1 (identity epilogue)", "beta": "0 (C not read)"},
    "cublaslt": {
        "input_dtype": {"bf16": "CUDA_R_16BF", "fp8": "CUDA_R_8F_E4M3", "nvfp4": "CUDA_R_4F_E2M1"},
        "scale_dtype": {"nvfp4": "CUDA_R_8F_UE4M3 (VEC16_UE4M3)"},
        "scale_layout": {"nvfp4": "documented 1D block-scaling layout; 128 rows x 4 blocks "
                                  "per 512 B; K blocks fastest"},
        "accumulator_dtype": "CUBLAS_COMPUTE_32F", "output_dtype": "CUDA_R_32F",
        "alpha": "1.0", "beta": "0.0"},
}


def arithmetic(implementation, precision):
    record = ARITHMETIC[implementation]
    scaled = precision == "nvfp4"
    return {"input_dtype": record["input_dtype"][precision],
            "scale_dtype": record["scale_dtype"].get(precision, ""),
            "scale_block_elements": SCALE_BLOCK if scaled else "",
            "scale_layout": record["scale_layout"].get(precision, ""),
            "accumulator_dtype": record["accumulator_dtype"],
            "output_dtype": record["output_dtype"],
            "alpha": record["alpha"], "beta": record["beta"]}


class PlanInfo(ctypes.Structure):
    """Mirror of GbpPlanInfo in cublaslt_precision_bridge.cu."""

    _fields_ = [("returned_algorithms", ctypes.c_int32), ("selected_index", ctypes.c_int32),
                ("algorithm_id", ctypes.c_int32), ("tile_id", ctypes.c_uint32),
                ("stages_id", ctypes.c_uint32), ("splitk_num", ctypes.c_int32),
                ("reduction_scheme", ctypes.c_uint32), ("cta_swizzling", ctypes.c_uint32),
                ("custom_option", ctypes.c_uint32), ("inner_shape_id", ctypes.c_uint16),
                ("cluster_shape_id", ctypes.c_uint16), ("workspace_bytes", ctypes.c_uint64),
                ("waves_count", ctypes.c_float), ("check_status", ctypes.c_int32),
                ("check_workspace_bytes", ctypes.c_uint64),
                ("algorithm_data", ctypes.c_uint64 * 8)]

    def record(self):
        values = {name: getattr(self, name) for name, _ in self._fields_ if name != "algorithm_data"}
        values["algorithm_data"] = "".join(f"{word:016x}" for word in self.algorithm_data)
        return values


class Plan:
    def __init__(self, library, handle, info):
        self.library, self.handle, self.info = library, handle, info

    def __call__(self):
        if self.library.gbp_plan_execute(self.handle):
            raise RuntimeError(self.library.gbp_last_error().decode())

    def close(self):
        if self.handle.value:
            self.library.gbp_plan_destroy(self.handle)
            self.handle = ctypes.c_void_p()


class Bridge:
    def __init__(self):
        self.library = ctypes.CDLL(str(BRIDGE_LIBRARY))
        self.library.gbp_last_error.restype = ctypes.c_char_p
        self.library.gbp_cublaslt_version.restype = ctypes.c_size_t
        self.library.gbp_plan_create.argtypes = [
            ctypes.c_int32, ctypes.c_int32, *[ctypes.c_int64] * 3, *[ctypes.c_void_p] * 6,
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(PlanInfo)]
        self.library.gbp_plan_execute.argtypes = [ctypes.c_void_p]
        self.library.gbp_plan_destroy.argtypes = [ctypes.c_void_p]

    def version(self):
        return int(self.library.gbp_cublaslt_version())

    def plan(self, precision, encoded, output, stream):
        m, n = output.shape
        k = encoded["a"].shape[1] * (2 if precision == "nvfp4" else 1)
        pointer = lambda name: encoded[name].data_ptr() if encoded.get(name) is not None else None
        handle, info = ctypes.c_void_p(), PlanInfo()
        if self.library.gbp_plan_create(
                FORMAT_CODES[precision], OUTPUT_FLOAT32, m, n, k, pointer("a"), pointer("b"),
                pointer("a_scale"), pointer("b_scale"), output.data_ptr(), stream,
                WORKSPACE_LIMIT_BYTES, ctypes.byref(handle), ctypes.byref(info)):
            raise RuntimeError(f"cuBLASLt {precision} FP32-output plan: "
                               f"{self.library.gbp_last_error().decode()} "
                               f"({info.returned_algorithms} heuristic results)")
        return Plan(self.library, handle, info)


def storage_bytes(torch, tensor, count):
    """Return count bytes of tensor's storage from its storage offset, in memory order."""
    raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(tensor.untyped_storage())
    start = tensor.storage_offset() * tensor.element_size()
    if start + count > raw.numel():
        raise ValueError("the captured tensor holds fewer bytes than its logical operand")
    return raw[start:start + count]


def fp4_pack(torch, values):
    """Encode exactly representable E2M1 values, two per byte with element 2i in the low nibble."""
    table = torch.tensor(E2M1_MAGNITUDES, device=values.device)
    magnitude = values.abs().contiguous()
    codes = torch.searchsorted(table, magnitude).clamp(max=len(E2M1_MAGNITUDES) - 1)
    if not bool((table[codes] == magnitude).all()):
        raise ValueError("an operand value is not representable in E2M1")
    codes = (codes | ((values < 0).to(codes.dtype) << 3)).to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def fp4_unpack(torch, packed):
    magnitudes = torch.tensor(E2M1_MAGNITUDES, device=packed.device)
    table = torch.cat((magnitudes, -magnitudes))
    values = torch.stack((table[(packed & 0xF).long()], table[(packed >> 4).long()]), dim=-1)
    return values.flatten(-2)


def e4m3_encode(torch, values):
    encoded = values.to(torch.float8_e4m3fn)
    if not bool((encoded.float() == values).all()):
        raise ValueError("an operand value is not representable in E4M3")
    return encoded.view(torch.uint8)


def e4m3_decode(torch, codes):
    return codes.view(torch.float8_e4m3fn).float()


def scale_offsets(torch, rows, blocks, device):
    """Byte offset of scale (row, block) in cuBLASLt's documented VEC16 block-scaling layout.

    128-row by 4-block tiles occupy 512 contiguous bytes, tiles advance along K first, and
    inside a tile row r and block c sit at (r % 32) * 16 + (r // 32) * 4 + c.
    """
    row = torch.arange(rows, device=device).unsqueeze(1)
    block = torch.arange(blocks, device=device).unsqueeze(0)
    tile = (row // 128) * math.ceil(blocks / 4) + block // 4
    return tile * 512 + (row % 32) * 16 + (row % 128) // 32 * 4 + block % 4


def scale_bytes(rows, blocks):
    return math.ceil(rows / 128) * 128 * math.ceil(blocks / 4) * 4


def scale_pack(torch, scales):
    rows, blocks = scales.shape
    packed = torch.zeros(scale_bytes(rows, blocks), dtype=torch.uint8, device=scales.device)
    packed[scale_offsets(torch, rows, blocks, scales.device).flatten()] = \
        e4m3_encode(torch, scales).flatten()
    return packed


def scale_unpack(torch, packed, rows, blocks):
    return e4m3_decode(torch, packed[scale_offsets(torch, rows, blocks, packed.device)])


def encode(torch, precision, operands):
    """Encode logical operands for cuBLASLt; every value must be represented exactly."""
    if precision == "nvfp4":
        return {"a": fp4_pack(torch, operands["a"]), "b": fp4_pack(torch, operands["b"]),
                "a_scale": scale_pack(torch, operands["a_scale"]),
                "b_scale": scale_pack(torch, operands["b_scale"])}
    if precision == "fp8":
        return {"a": e4m3_encode(torch, operands["a"]), "b": e4m3_encode(torch, operands["b"])}
    encoded = {"a": operands["a"].to(torch.bfloat16), "b": operands["b"].to(torch.bfloat16)}
    if not all(bool((encoded[name].float() == operands[name]).all()) for name in ("a", "b")):
        raise ValueError("an operand value is not representable in BF16")
    return encoded


def decode(torch, precision, encoded, shape):
    """Dequantize the bytes cuBLASLt consumes back to FP32 operands."""
    m, n, k, _ = shape
    if precision == "nvfp4":
        blocks = k // SCALE_BLOCK
        return tuple(fp4_unpack(torch, encoded[name]) * scale_unpack(
            torch, encoded[f"{name}_scale"], rows, blocks).repeat_interleave(SCALE_BLOCK, dim=1)
            for name, rows in (("a", m), ("b", n)))
    if precision == "fp8":
        return e4m3_decode(torch, encoded["a"]), e4m3_decode(torch, encoded["b"])
    return encoded["a"].float(), encoded["b"].float()


def logical_dequantized(torch, precision, operands, name):
    if precision != "nvfp4":
        return operands[name]
    return operands[name] * operands[f"{name}_scale"].repeat_interleave(SCALE_BLOCK, dim=1)


def reference(torch, a, b):
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "ieee"
        return a @ b.T
    finally:
        matmul.fp32_precision = previous


def compare(torch, precision, output, expected):
    atol, rtol = TOLERANCES[precision]
    finite = bool(torch.isfinite(output).all())
    difference = (output - expected).abs()
    mismatches = int((difference > atol + rtol * expected.abs()).sum().item())
    return {"status": "PASS" if finite and mismatches == 0 else "FAIL", "atol": atol,
            "rtol": rtol, "finite": finite, "mismatches": mismatches,
            "max_abs_error": float(difference.max().item()),
            "bit_exact": bool(torch.equal(output, expected)),
            # Nonzero counts with a bit-exact result show the output was not rounded to BF16/FP16.
            "reference_values_not_bf16_exact": int(
                (expected != expected.to(torch.bfloat16).float()).sum().item()),
            "reference_values_not_fp16_exact": int(
                (expected != expected.to(torch.float16).float()).sum().item())}


def check_validator(torch):
    """Negative control: every format's check must reject one perturbed value and pass a copy."""
    expected = torch.full((4, 4), 100.0)
    perturbed = expected.clone()
    perturbed[1, 2] += 2  # beyond atol + rtol * 100 for every format (at most 1.1)
    for precision in TOLERANCES:
        if (compare(torch, precision, perturbed, expected)["mismatches"] != 1 or
                compare(torch, precision, expected.clone(), expected)["status"] != "PASS"):
            raise RuntimeError(f"the {precision} validation check does not reject a mismatch")


@contextlib.contextmanager
def capture_dense(module):
    """Record each operand set that the dense example's run() prepares."""
    calls, original = [], module.prepare_tensors

    def recorder(*arguments, **keywords):
        result = original(*arguments, **keywords)
        calls.append(result)
        return result

    module.prepare_tensors = recorder
    try:
        yield calls
    finally:
        module.prepare_tensors = original


@contextlib.contextmanager
def capture_blockscaled():
    """Record the NVFP4 example's logical tensors and the device tensors it converts them to."""
    import cutlass.torch as cutlass_torch

    created, converted = [], []
    original_create = cutlass_torch.create_and_permute_torch_tensor
    original_like = cutlass_torch.cute_tensor_like
    signature = inspect.signature(original_create)

    def create(*arguments, **keywords):
        bound = signature.bind(*arguments, **keywords)
        bound.apply_defaults()
        result = original_create(*arguments, **keywords)
        created.append({**bound.arguments, "tensor": result})
        return result

    def like(data_ref, cutlass_dtype, *arguments, **keywords):
        result = original_like(data_ref, cutlass_dtype, *arguments, **keywords)
        converted.append({"dtype": cutlass_dtype, "tensor": result[1]})
        return result

    cutlass_torch.create_and_permute_torch_tensor = create
    cutlass_torch.cute_tensor_like = like
    try:
        yield {"created": created, "converted": converted}
    finally:
        cutlass_torch.create_and_permute_torch_tensor = original_create
        cutlass_torch.cute_tensor_like = original_like


def dense_sets(torch, precision, shape, calls):
    """Return the first (validated in repetition 1) and the timed operand set of one run()."""
    if len(calls) != 2:
        raise RuntimeError(f"the dense example prepared {len(calls)} operand sets, expected 2")
    m, n, k, _ = shape
    width = 2 if precision == "bf16" else 1
    sets = []
    for a_f32, b_f32, _, a_storage, b_storage, c_storage in calls:
        # a_f32 is (L,M,K); b_f32 is an (L,K,N) view of N×K memory, so its transpose is K-major.
        sets.append({"a": a_f32[0], "b": b_f32[0].transpose(0, 1),
                     "cute": {"a": storage_bytes(torch, a_storage, m * k * width),
                              "b": storage_bytes(torch, b_storage, n * k * width),
                              "output": c_storage[0]}})
    return sets


def blockscaled_sets(torch, cutlass, shape, capture):
    """Return the validated and the timed NVFP4 operand set of one run()."""
    from cutlass.torch import TensorInitType

    m, n, k, batch = shape
    blocks = k // SCALE_BLOCK
    random = [entry for entry in capture["created"] if entry["init_type"] == TensorInitType.RANDOM]
    converted = capture["converted"]
    expected_shapes = [(batch, m, k), (batch, n, k), (batch, m, n)] + \
        [(batch, rows, blocks) for rows in (m, n, m, n)]
    expected_types = [cutlass.Float4E2M1FN, cutlass.Float4E2M1FN, cutlass.Float32,
                      cutlass.Float8E4M3FN, cutlass.Float8E4M3FN] * 2
    if ([tuple(entry["shape"]) for entry in random] != expected_shapes or
            [entry["dtype"] for entry in converted] != expected_types):
        raise RuntimeError("the NVFP4 example created its operands in an unexpected order")
    # Both sets reuse the same A and B values; only their scale factors are drawn anew.
    a, b = random[0]["tensor"][:, :, 0].cuda(), random[1]["tensor"][:, :, 0].cuda()
    sets = []
    for scales, tensors in ((random[3:5], converted[:5]), (random[5:7], converted[5:])):
        sets.append({
            "a": a, "b": b, "a_scale": scales[0]["tensor"][:, :, 0].cuda(),
            "b_scale": scales[1]["tensor"][:, :, 0].cuda(),
            "cute": {"a": storage_bytes(torch, tensors[0]["tensor"], m * k // 2),
                     "b": storage_bytes(torch, tensors[1]["tensor"], n * k // 2),
                     "a_scale": storage_bytes(torch, tensors[3]["tensor"], scale_bytes(m, blocks)),
                     "b_scale": storage_bytes(torch, tensors[4]["tensor"], scale_bytes(n, blocks)),
                     "output": tensors[2]["tensor"][:, :, 0]}})
    return sets


class Baseline:
    """Measure cuBLASLt on the operand sets captured from each CuTe DSL configuration."""

    def __init__(self, warmup, iterations):
        import torch
        from cuda.bindings import driver

        check_validator(torch)
        self.torch, self.driver = torch, driver
        self.warmup, self.iterations = warmup, iterations
        self.bridge = Bridge()
        self.repetitions, self.validation, self.operands, self.plans = [], [], [], []
        self.limitations = []

    @contextlib.contextmanager
    def capture(self, module, precision):
        with (capture_blockscaled() if precision == "nvfp4" else capture_dense(module)) as calls:
            yield calls

    def operand_sets(self, cutlass, precision, shape, captures):
        split = [blockscaled_sets(self.torch, cutlass, shape, capture) if precision == "nvfp4"
                 else dense_sets(self.torch, precision, shape, capture) for capture in captures]
        # Repetition 1 validates its first set; every repetition times its second set.
        return [("validated", split[0][0])] + [(f"timed_{index}", sets[1])
                                               for index, sets in enumerate(split, 1)]

    def check(self, context, operand_set, implementation, stage, output, expected, reference_kind):
        record = {**context, "operand_set": operand_set, "implementation": implementation,
                  "stage": stage, "reference": reference_kind,
                  **compare(self.torch, context["precision"], output, expected)}
        self.validation.append(record)
        return record

    def measure(self, shape_index, shape, precision, cutlass, captures, cute_samples):
        """Validate and time cuBLASLt on the CuTe DSL run's own logical operand sets."""
        torch = self.torch
        m, n, k, batch = shape
        flops = 2 * m * n * k * batch
        context = {"shape_index": shape_index, "shape_id": "x".join(map(str, shape)),
                   "m": m, "n": n, "k": k, "l": batch, "precision": precision}
        stream = torch.cuda.current_stream()
        plans = []
        for name, operands in self.operand_sets(cutlass, precision, shape, captures):
            repetition = int(name.split("_")[1]) if name.startswith("timed_") else None
            encoded = encode(torch, precision, operands)
            a, b = decode(torch, precision, encoded, shape)
            expected = reference(torch, a, b)
            digest = hashlib.sha256()
            for operand in encoded.values():
                digest.update(operand.contiguous().view(torch.uint8).cpu().numpy())
            self.operands.append({**context, "operand_set": name,
                                  "operands_sha256": digest.hexdigest(), **{
                f"{operand}_bytes_identical_to_cutedsl": bool(torch.equal(
                    encoded[operand].contiguous().view(torch.uint8).flatten(),
                    operands["cute"][operand])) for operand in encoded},
                "represented_exactly": all(bool(torch.equal(decoded, logical_dequantized(
                    torch, precision, operands, operand))) for operand, decoded in (("a", a),
                                                                                  ("b", b)))})
            # The example checked only its first launch; every output it left behind, including
            # the timed repetitions' results, must also meet this reference.
            cute = self.check(context, name, "cutedsl", "after_example_run",
                              operands["cute"]["output"], expected, REFERENCE)
            if repetition:
                self.repetitions.append({
                    **context, "implementation": "cutedsl", "repetition": repetition,
                    "operand_set": name, "kernel_time_us": cute_samples[repetition - 1],
                    "tflops": flops / cute_samples[repetition - 1] / 1e6,
                    "validation": cute["status"], "bit_exact": cute["bit_exact"]})
            output = torch.full((m, n), float("nan"), dtype=torch.float32, device="cuda")
            try:
                plan = self.bridge.plan(precision, encoded, output, stream.cuda_stream)
            except RuntimeError as error:
                self.limitations.append({**context, "operand_set": name, "reason": str(error)})
                return
            try:
                plans.append({"operand_set": name, **plan.info.record()})
                plan()
                torch.cuda.synchronize()
                before = self.check(context, name, "cublaslt", "before_timing", output,
                                    expected, REFERENCE)
                if before["status"] != "PASS":
                    raise RuntimeError(f"{context['shape_id']}/{precision}/{name}: the cuBLASLt "
                                       f"result failed validation")
                if repetition:
                    duration = self.time(plan)
                    after = self.check(context, name, "cublaslt", "after_timing", output,
                                       expected, REFERENCE)
                    self.repetitions.append({
                        **context, "implementation": "cublaslt", "repetition": repetition,
                        "operand_set": name, "kernel_time_us": duration,
                        "tflops": flops / duration / 1e6, "validation": after["status"],
                        "bit_exact": before["bit_exact"] and after["bit_exact"]})
                self.operands[-1]["outputs_bit_identical"] = bool(
                    torch.equal(output, operands["cute"]["output"]))
            finally:
                plan.close()
            del encoded, a, b, expected, output
        identical = len({plan["algorithm_data"] for plan in plans}) == 1
        self.plans.append({**context, "plans": plans, "same_algorithm_for_every_set": identical,
                           "algorithm": {key: plans[0][key] for key in plans[0]
                                         if key != "operand_set"}})
        torch.cuda.synchronize()

    def time(self, plan):
        """Time exactly like the CuTe DSL examples: cute.testing.benchmark on the default stream."""
        from cutlass.cute import testing

        stream = self.driver.CUstream(self.torch.cuda.current_stream().cuda_stream)
        microseconds = float(testing.benchmark(
            plan, kernel_arguments=testing.JitArguments(), stream=stream,
            warmup_iterations=self.warmup, iterations=self.iterations))
        if not math.isfinite(microseconds) or microseconds <= 0:
            raise RuntimeError(f"invalid cuBLASLt kernel time {microseconds}")
        return microseconds

    def identify_kernels(self):
        """Name the kernel behind every selected algorithm once all timing is finished."""
        torch = self.torch
        # torch.profiler imports Inductor, whose default cache path needs a passwd entry.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
        from torch.profiler import ProfilerActivity, profile

        for entry in self.plans:
            m, n, k, precision = entry["m"], entry["n"], entry["k"], entry["precision"]
            if precision == "nvfp4":
                blocks = k // SCALE_BLOCK
                encoded = {"a": torch.zeros(m, k // 2, dtype=torch.uint8, device="cuda"),
                           "b": torch.zeros(n, k // 2, dtype=torch.uint8, device="cuda"),
                           "a_scale": torch.zeros(scale_bytes(m, blocks), dtype=torch.uint8,
                                                  device="cuda"),
                           "b_scale": torch.zeros(scale_bytes(n, blocks), dtype=torch.uint8,
                                                  device="cuda")}
            else:
                dtype = torch.bfloat16 if precision == "bf16" else torch.uint8
                encoded = {"a": torch.zeros(m, k, dtype=dtype, device="cuda"),
                           "b": torch.zeros(n, k, dtype=dtype, device="cuda")}
            output = torch.empty(m, n, dtype=torch.float32, device="cuda")
            plan = self.bridge.plan(precision, encoded, output,
                                    torch.cuda.current_stream().cuda_stream)
            try:
                entry["identification_algorithm_matches"] = (
                    plan.info.record()["algorithm_data"] == entry["algorithm"]["algorithm_data"])
                plan()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as trace:
                    plan()
                    torch.cuda.synchronize()
            finally:
                plan.close()
            names = [event.name for event in trace.events() if event.device_type.name == "CUDA"
                     and not event.name.startswith(("Memset", "Memcpy"))]
            entry["kernel_names"] = names
            entry["kernels_per_launch"] = len(names)
