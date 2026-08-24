#!/usr/bin/env python3
"""Run the five frozen GEMM shapes with three CuTe DSL variants and cuBLASLt.

All candidates share one operand set and one untimed FP32 reference per shape.
Correctness is mandatory before warm-up and CUDA-event timing. The program
emits one complete 20-row CSV only after every candidate passes; diagnostics
go to stderr. Only the warm-up and measured iteration counts are configurable.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- Frozen identity ---------------------------------------------------------

SCHEMA_VERSION = "p35.v1"
EXPERIMENT = "exp03_cutedsl_vs_cublaslt"
UNIT = "P3.5"
RUN_KIND = "smoke"
REFERENCE = "torch_cuda_fp32_ieee"
CACHE_MODE = "hot"
CORRECTNESS_PASS = "PASS"
PUBLISHABLE = "false"

# The canonical value recorded for a field that has no meaning for the row's
# method or variant. It is never a number, never zero, and never empty.
NOT_APPLICABLE = "not_applicable"

# --- Frozen shape table ------------------------------------------------------
#
# The five and only five (M,N,K,L) shapes of Experiment 3, in this exact order.
# None of them is reachable from the command line, from an environment
# variable, from a configuration file, or from an input CSV. The compiled
# cuBLASLt bridge freezes the same five independently in C, and the two
# allowlists are compared at run time.

FROZEN_L = 1

FROZEN_SHAPES = (
    (4096, 4096, 4096, 1),
    (8192, 8192, 8192, 1),
    (16384, 512, 4096, 1),
    (32768, 512, 4096, 1),
    (512, 16384, 4096, 1),
)

FROZEN_SHAPE_COUNT = len(FROZEN_SHAPES)


def shape_id(mnkl) -> str:
    """The canonical single-token identifier of one frozen shape."""
    m, n, k, l = mnkl
    return f"{m}x{n}x{k}x{l}"


FROZEN_SHAPE_IDS = tuple(shape_id(mnkl) for mnkl in FROZEN_SHAPES)

# --- Frozen GEMM contract ----------------------------------------------------

FROZEN_AB_DTYPE = "BFloat16"
FROZEN_ACC_DTYPE = "Float32"
FROZEN_C_DTYPE = "Float32"

FROZEN_A_MAJOR = "k"
FROZEN_B_MAJOR = "k"
FROZEN_C_MAJOR = "n"

FROZEN_USE_TMA_STORE = True

FROZEN_SEED = 1111
FROZEN_ATOL = 1e-1
FROZEN_RTOL = 1e-5

# --- Frozen cuBLASLt descriptor and algorithm policy ------------------------

FROZEN_TRANSA = "CUBLAS_OP_N"
FROZEN_TRANSB = "CUBLAS_OP_T"
FROZEN_ORDER = "CUBLASLT_ORDER_ROW"
FROZEN_AB_CUDA_TYPE = "CUDA_R_16BF"
FROZEN_CD_CUDA_TYPE = "CUDA_R_32F"
FROZEN_COMPUTE_TYPE = "CUBLAS_COMPUTE_32F"
FROZEN_SCALE_TYPE = "CUDA_R_32F"
FROZEN_POINTER_MODE = "CUBLASLT_POINTER_MODE_HOST"
FROZEN_EPILOGUE = "CUBLASLT_EPILOGUE_DEFAULT"
FROZEN_SEARCH_MODE = "CUBLASLT_SEARCH_BEST_FIT"

FROZEN_ALPHA = 1.0
FROZEN_BETA = 0.0

FROZEN_WORKSPACE_LIMIT_BYTES = 67108864
FROZEN_HEURISTIC_REQUESTED = 32

# The numeric values of every frozen cuBLASLt enum, as declared in the pinned
# CUDA 13.1 headers. The bridge reports the values it actually compiled
# against; these are the wrapper's independent expectation, and a disagreement
# fails closed instead of being silently serialized.
CUBLAS_ENUM_VALUES = {
    "CUBLAS_OP_N": 0,
    "CUBLAS_OP_T": 1,
    "CUBLASLT_ORDER_ROW": 1,
    "CUDA_R_16BF": 14,
    "CUDA_R_32F": 0,
    "CUBLAS_COMPUTE_32F": 68,
    "CUBLASLT_POINTER_MODE_HOST": 0,
    "CUBLASLT_EPILOGUE_DEFAULT": 1,
    "CUBLASLT_SEARCH_BEST_FIT": 0,
}

# --- Frozen candidate table --------------------------------------------------
#
# Exactly four candidates, always executed in this order for every shape. This
# is not a search space: there is no autotuning, no ranking-driven selection,
# and no fifth candidate.

METHOD_CUTEDSL = "cutedsl"
METHOD_CUBLASLT = "cublaslt"

VARIANT_NONPERSISTENT_1CTA = "nonpersistent_1cta"
VARIANT_PERSISTENT_1CTA = "persistent_1cta"
VARIANT_PERSISTENT_2CTA = "persistent_2cta"
VARIANT_CUBLASLT = "heuristic_first_supported"

SCHEDULER_NONPERSISTENT = "nonpersistent"
SCHEDULER_STATIC_PERSISTENT = "static_persistent"

SOURCE_NONPERSISTENT = "nonpersistent"
SOURCE_PERSISTENT = "persistent"

UPSTREAM_CLASS_NONPERSISTENT = "DenseGemmKernel"
UPSTREAM_CLASS_PERSISTENT = "PersistentDenseGemmKernel"

FROZEN_CANDIDATES = (
    {
        "method": METHOD_CUTEDSL,
        "variant": VARIANT_NONPERSISTENT_1CTA,
        "scheduler": SCHEDULER_NONPERSISTENT,
        "source": SOURCE_NONPERSISTENT,
        "upstream_class": UPSTREAM_CLASS_NONPERSISTENT,
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "use_2cta_instrs": False,
        "persistent": False,
    },
    {
        "method": METHOD_CUTEDSL,
        "variant": VARIANT_PERSISTENT_1CTA,
        "scheduler": SCHEDULER_STATIC_PERSISTENT,
        "source": SOURCE_PERSISTENT,
        "upstream_class": UPSTREAM_CLASS_PERSISTENT,
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "use_2cta_instrs": False,
        "persistent": True,
    },
    {
        # M tile 256 with a 2-CTA cluster keeps the per-CTA M extent at 128,
        # matching the 2-SM UMMA geometry and NVIDIA's own documented 2-CTA
        # constraint. Never substituted.
        "method": METHOD_CUTEDSL,
        "variant": VARIANT_PERSISTENT_2CTA,
        "scheduler": SCHEDULER_STATIC_PERSISTENT,
        "source": SOURCE_PERSISTENT,
        "upstream_class": UPSTREAM_CLASS_PERSISTENT,
        "mma_tiler_mn": (256, 128),
        "cluster_shape_mn": (2, 1),
        "use_2cta_instrs": True,
        "persistent": True,
    },
    {
        # The comparison baseline. Its selection policy never changes;
        # only which supported algorithm the vendor heuristic
        # happens to return may differ per shape.
        "method": METHOD_CUBLASLT,
        "variant": VARIANT_CUBLASLT,
        "scheduler": None,
        "source": None,
        "upstream_class": None,
        "mma_tiler_mn": None,
        "cluster_shape_mn": None,
        "use_2cta_instrs": None,
        "persistent": False,
    },
)

FROZEN_CANDIDATE_COUNT = len(FROZEN_CANDIDATES)
FROZEN_CANDIDATE_ORDER = tuple(spec["variant"] for spec in FROZEN_CANDIDATES)

# The index of the comparison baseline and of the three CuTe DSL candidates.
CUBLASLT_CANDIDATE_INDEX = 3
CUTEDSL_CANDIDATE_INDICES = (0, 1, 2)

EXPECTED_ROW_COUNT = FROZEN_SHAPE_COUNT * FROZEN_CANDIDATE_COUNT  # 20

# The only CUDA matmul FP32 policy accepted for the correctness oracle,
# via the PyTorch 2.10 fp32_precision API and nothing else. The unset default
# is "none", which proves nothing and is rejected.
FP32_PRECISION_IEEE = "ieee"

# Safe denominator for the reported relative error.
REL_ERROR_DENOMINATOR_FLOOR = 1.0

# The factor of two in flop_count counts one multiplication plus one addition
# per multiply-accumulate. Nothing else - no setup, compilation, first launch,
# correctness, reset, or epilogue bookkeeping - is counted.
FLOPS_PER_MAC = 2

FROZEN_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "shapes": FROZEN_SHAPES,
    "ab_dtype": FROZEN_AB_DTYPE,
    "acc_dtype": FROZEN_ACC_DTYPE,
    "c_dtype": FROZEN_C_DTYPE,
    "a_major": FROZEN_A_MAJOR,
    "b_major": FROZEN_B_MAJOR,
    "c_major": FROZEN_C_MAJOR,
    "use_tma_store": FROZEN_USE_TMA_STORE,
    "seed": FROZEN_SEED,
    "reference": REFERENCE,
    "atol": FROZEN_ATOL,
    "rtol": FROZEN_RTOL,
    "cache_mode": CACHE_MODE,
    "workspace_limit_bytes": FROZEN_WORKSPACE_LIMIT_BYTES,
    "heuristic_requested": FROZEN_HEURISTIC_REQUESTED,
    "search_mode": FROZEN_SEARCH_MODE,
    "flops_per_mac": FLOPS_PER_MAC,
    "publishable": False,
}

# --- Runtime controls (the only ones that exist) -----------------------------

DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_ITERATIONS = 20
MIN_ITERATIONS = 1
MAX_WARMUP_ITERATIONS = 100
MAX_ITERATIONS = 100

# --- Frozen CSV schema -------------------------------------------------------

CSV_FIELDS = (
    # Unit, schema, experiment, and run identity.
    "schema_version",
    "experiment",
    "unit",
    "run_kind",
    # Shape and candidate identity.
    "shape_index",
    "shape_id",
    "candidate_index",
    "method",
    "variant",
    # The problem.
    "m",
    "n",
    "k",
    "l",
    "ab_dtype",
    "acc_dtype",
    "c_dtype",
    "a_major",
    "b_major",
    "c_major",
    # CuTe DSL metadata (not_applicable on the cuBLASLt row).
    "scheduler",
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
    "max_active_clusters",
    # cuBLASLt metadata (not_applicable on the three CuTe DSL rows).
    "order_a",
    "order_b",
    "order_c",
    "order_d",
    "transa",
    "transb",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "compute_type",
    "scale_type",
    "pointer_mode",
    "epilogue",
    "alpha",
    "beta",
    "search_mode",
    "workspace_limit_bytes",
    "workspace_bytes",
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
    "heuristic_requested",
    "heuristic_returned",
    "heuristic_index",
    "algo_id",
    "tile_id",
    "stages_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzling",
    "custom_option",
    "inner_shape_id",
    "cluster_shape_id",
    "waves_count",
    "cublaslt_version",
    # Correctness.
    "seed",
    "reference",
    "atol",
    "rtol",
    "correctness",
    "max_abs_error",
    "max_rel_error",
    # Timing boundaries.
    "compile_time_ms",
    "setup_time_ms",
    "first_launch_ms",
    "kernel_time_ms",
    "warmup_iterations",
    "iterations",
    "cache_mode",
    # The comparison.
    "flop_count",
    "tflops",
    "throughput_ratio_vs_cublaslt",
    "gap_to_cublaslt_pct",
    "rank_within_shape",
    "best_cutedsl_variant",
    "is_best_cutedsl",
    # Provenance.
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_toolkit_version",
    "torch_cuda_version",
    "cutedsl_version",
    "cutlass_commit",
    "operand_factory_sha256",
    "upstream_kernel_file",
    "upstream_kernel_git_blob",
    "upstream_kernel_sha256",
    "git_commit",
    "git_dirty",
    "publishable",
)

# Deterministic decimal formats. Every real-valued field is serialized as a
# plain fixed-point decimal with exactly this many fractional digits: no
# exponent, no locale dependence, no shortest-round-trip ambiguity. Every
# decision - correctness, positivity, finiteness, ranking, best-variant
# selection - is taken on the full-precision value before serialization.
DECIMALS_TIMING = 6  # milliseconds, i.e. nanosecond resolution
DECIMALS_ERROR = 9
DECIMALS_TOLERANCE = 9
DECIMALS_SCALAR = 9  # alpha and beta
DECIMALS_WAVES = 6
DECIMALS_TFLOPS = 6
DECIMALS_RATIO = 9
DECIMALS_GAP = 6  # signed: a negative gap means the candidate is faster

# Values identical in all twenty rows.
CSV_FIXED_VALUES = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "l": str(FROZEN_L),
    "ab_dtype": FROZEN_AB_DTYPE,
    "acc_dtype": FROZEN_ACC_DTYPE,
    "c_dtype": FROZEN_C_DTYPE,
    "a_major": FROZEN_A_MAJOR,
    "b_major": FROZEN_B_MAJOR,
    "c_major": FROZEN_C_MAJOR,
    "seed": str(FROZEN_SEED),
    "reference": REFERENCE,
    "correctness": CORRECTNESS_PASS,
    "cache_mode": CACHE_MODE,
    "publishable": PUBLISHABLE,
}

# Fields that carry a value only on a CuTe DSL row.
CUTEDSL_ONLY_FIELDS = (
    "scheduler",
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
    "max_active_clusters",
    "compile_time_ms",
    "upstream_kernel_file",
    "upstream_kernel_git_blob",
    "upstream_kernel_sha256",
)

# Fields that carry a value only on the cuBLASLt row.
CUBLASLT_ONLY_FIELDS = (
    "order_a",
    "order_b",
    "order_c",
    "order_d",
    "transa",
    "transb",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "compute_type",
    "scale_type",
    "pointer_mode",
    "epilogue",
    "alpha",
    "beta",
    "search_mode",
    "workspace_limit_bytes",
    "workspace_bytes",
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
    "heuristic_requested",
    "heuristic_returned",
    "heuristic_index",
    "algo_id",
    "tile_id",
    "stages_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzling",
    "custom_option",
    "inner_shape_id",
    "cluster_shape_id",
    "waves_count",
    "cublaslt_version",
    "setup_time_ms",
)

CSV_ERROR_FIELDS = ("max_abs_error", "max_rel_error")
CSV_TOLERANCE_FIELDS = ("atol", "rtol")
CSV_COUNT_FIELDS = ("warmup_iterations", "iterations")
CSV_BOOL_FIELDS = ("git_dirty", "publishable")
# Booleans that are canonical true/false on their applicable rows only.
CSV_METHOD_BOOL_FIELDS = ("use_2cta_instrs", "use_tma_store")

BOOL_TRUE = "true"
BOOL_FALSE = "false"

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_NONNEGATIVE_INT = re.compile(r"\A(0|[1-9][0-9]*)\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_CUDA_ARCH = re.compile(r"\Asm_([0-9]+)([a-z]?)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")
_RE_SHAPE_ID = re.compile(r"\A[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*\Z")
# A repository-relative upstream path: must start with an alphanumeric (so an
# absolute path is rejected) and must contain no ".." segment.
_RE_UPSTREAM_REL_PATH = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\.py\Z")


def is_relative_upstream_path(path: str) -> bool:
    """True only for a safe, repository-relative upstream ``.py`` path."""
    if not isinstance(path, str) or not _RE_UPSTREAM_REL_PATH.match(path):
        return False
    return ".." not in Path(path).parts


# The pinned CUTLASS checkout inside the pinned image. The image builds it at
# exactly CUTLASS_COMMIT (see Dockerfile); nothing is ever written to it, and
# this location is not configurable at runtime.
UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

# The compiled cuBLASLt bridge, built into the container's own private /tmp by
# the Make targets immediately before the wrapper runs. This location is a
# constant, not a runtime control: there is no CLI option and no environment
# variable for it.
BRIDGE_LIBRARY_PATH = Path("/tmp/p35-bridge/libp35_cublaslt_bridge.so")
BRIDGE_SOURCE_RELATIVE = "gemm_comparison/cublaslt_bridge.cu"
BRIDGE_ABI_VERSION = 1

GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# Keys read from the two version contracts. Nothing below is duplicated as a
# literal anywhere in this file: the pinned commit, blobs, SHA-256 digests,
# versions, and architecture exist here only as key names. No key is added to
# either contract - this reuses the two already pinned official sources and the
# cuBLASLt library that already ships in the pinned CUDA 13.1 image.
GLOBAL_CONTRACT_KEYS = ("CUDA_VERSION", "CUTLASS_VERSION", "CUTLASS_COMMIT", "CUDA_ARCH")
PHASE3_CONTRACT_KEYS = (
    "PYTORCH_VERSION",
    "PYTORCH_CUDA_VERSION",
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
)

# The two pinned upstream sources, keyed by the source name each CuTe DSL
# candidate names.
UPSTREAM_SOURCES = {
    SOURCE_NONPERSISTENT: {
        "path_key": "CUTEDSL_P31_EXAMPLE_PATH",
        "blob_key": "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
        "sha256_key": "CUTEDSL_P31_EXAMPLE_SHA256",
        "kernel_class": UPSTREAM_CLASS_NONPERSISTENT,
        "module_name": "p35_pinned_upstream_dense_gemm",
    },
    SOURCE_PERSISTENT: {
        "path_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
        "blob_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
        "sha256_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
        "kernel_class": UPSTREAM_CLASS_PERSISTENT,
        "module_name": "p35_pinned_upstream_dense_gemm_persistent",
    },
}

# The operand factory lives in the non-persistent example only. The persistent
# example's own tensor-generation path is deliberately never used, because that
# would break byte-for-byte operand equivalence across candidates.
OPERAND_FACTORY_SOURCE = SOURCE_NONPERSISTENT
OPERAND_FACTORY_NAME = "create_tensors"
UPSTREAM_MATRIX_FACTORY = "matrix"
# The exact upstream ``cutlass.torch.matrix`` call sequence, in call order, as
# (l, mode0, mode1, dtype) argument names. This code depends on the shapes and
# strides this sequence produces, because those are what its cuBLASLt leading
# dimensions assume, so a divergence must fail the run rather than silently
# change the descriptors.
UPSTREAM_MATRIX_CALLS = (
    ("l", "m", "k", "ab_dtype"),
    ("l", "n", "k", "ab_dtype"),
    ("l", "m", "n", "c_dtype"),
)


class P35Error(Exception):
    """Any fail-closed contract, provenance, or execution failure."""


class RowContractError(P35Error):
    """A CSV row violated the frozen schema."""


class CorrectnessError(P35Error):
    """A candidate's complete result did not match the untimed FP32 reference."""


class BridgeError(P35Error):
    """The cuBLASLt bridge refused, failed, or disagreed with this wrapper."""


class ComparisonError(P35Error):
    """A comparison quantity could not be computed deterministically."""


def log(message: str) -> None:
    """Write one human-readable progress/diagnostic line to stderr."""
    print(f"gemm_comparison: {message}", file=sys.stderr, flush=True)


def _cleanup_preserving_primary(cleanup, description: str) -> None:
    """Fail on cleanup after success without masking an active primary error.

    Python replaces an exception raised in a ``try`` block when its ``finally``
    block raises another one. Both halves of a stricter contract are needed: a
    cleanup failure after otherwise successful work invalidates the run, while
    a cleanup failure during an already failing operation is reported without
    hiding the operation's original diagnostic.
    """
    primary_error_active = sys.exc_info()[0] is not None
    try:
        cleanup()
    except BaseException as cleanup_error:  # noqa: BLE001 - preserve the active primary error
        if not primary_error_active:
            raise
        log(
            f"WARNING: {description} also failed while preserving the original error: "
            f"{cleanup_error}"
        )


# --- Version contracts -------------------------------------------------------


def repository_root() -> Path:
    """Locate the repository root that owns this file.

    ``gemm_comparison/gemm_comparison.py`` is one directory below the root both
    on the host and inside the container, where the repository is mounted at
    ``/workspace``.
    """
    root = Path(__file__).resolve().parents[1]
    for name in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        if not (root / name).is_file():
            raise P35Error(f"repository root {root} does not contain {name}")
    return root


def parse_env_file(path: Path) -> dict:
    """Parse a ``KEY=VALUE`` version contract strictly and fail closed."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P35Error(f"cannot read version contract {path}: {exc}") from exc

    values: dict = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise P35Error(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise P35Error(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


def load_pinned_contract(repo_root=None) -> dict:
    """Read every pinned value this program needs from the two contracts.

    ``VERSIONS.env`` is the global contract, read here but never written.
    ``PHASE3_VERSIONS.env`` holds the CuTe DSL / cuBLASLt pins. **No key is
    added to either file**: this reuses the two already pinned official
    examples and the cuBLASLt library that already ships inside the pinned CUDA
    13.1 image, whose runtime version is read with ``cublasLtGetVersion()``
    rather than pinned.
    """
    root = Path(repo_root) if repo_root is not None else repository_root()
    global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
    phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)

    contract = {}
    for key in GLOBAL_CONTRACT_KEYS:
        if key not in global_values:
            raise P35Error(f"{GLOBAL_CONTRACT_FILE} is missing required key {key}")
        contract[key] = global_values[key]
    for key in PHASE3_CONTRACT_KEYS:
        if key not in phase3_values:
            raise P35Error(f"{PHASE3_CONTRACT_FILE} is missing required key {key}")
        contract[key] = phase3_values[key]

    if not _RE_HEX40.match(contract["CUTLASS_COMMIT"]):
        raise P35Error(f"pinned CUTLASS_COMMIT is malformed: {contract['CUTLASS_COMMIT']!r}")
    for source in UPSTREAM_SOURCES.values():
        blob = contract[source["blob_key"]]
        sha256 = contract[source["sha256_key"]]
        path = contract[source["path_key"]]
        if not _RE_HEX40.match(blob):
            raise P35Error(f"pinned {source['blob_key']} is malformed")
        if not _RE_HEX64.match(sha256):
            raise P35Error(f"pinned {source['sha256_key']} is malformed")
        if not is_relative_upstream_path(path):
            raise P35Error(f"pinned {source['path_key']} is unsafe: {path!r}")

    non_persistent_path = contract[UPSTREAM_SOURCES[SOURCE_NONPERSISTENT]["path_key"]]
    persistent_path = contract[UPSTREAM_SOURCES[SOURCE_PERSISTENT]["path_key"]]
    if non_persistent_path == persistent_path:
        raise P35Error(
            "the pinned non-persistent and persistent examples are the same file; "
            "P3.5 requires two distinct official sources"
        )

    for key in ("CUDA_VERSION", "PYTORCH_CUDA_VERSION"):
        if not _RE_DOTTED_VERSION.match(contract[key]):
            raise P35Error(f"pinned {key} is malformed: {contract[key]!r}")
    if not contract["CUTLASS_VERSION"].startswith("v"):
        raise P35Error(f"pinned CUTLASS_VERSION is malformed: {contract['CUTLASS_VERSION']!r}")

    # Derived, never separately pinned.
    contract["CUTEDSL_VERSION"] = contract["CUTLASS_VERSION"][1:]
    if not _RE_DOTTED_VERSION.match(contract["CUTEDSL_VERSION"]):
        raise P35Error("pinned CuTe DSL version is malformed")

    contract["CUDA_MAJOR_MINOR"] = ".".join(contract["CUDA_VERSION"].split(".")[:2])
    contract["EXPECTED_COMPUTE_CAPABILITY"] = compute_capability_for_arch(contract["CUDA_ARCH"])
    return contract


def compute_capability_for_arch(cuda_arch: str) -> str:
    """Map a pinned ``sm_<digits>[a]`` target to its ``major.minor`` capability.

    NVIDIA's convention is that the final digit is the minor version and every
    preceding digit is the major one: ``sm_75`` is 7.5, ``sm_90`` is 9.0, and
    ``sm_100`` is 10.0. A trailing letter marks the architecture-specific form
    of the same capability. Deriving the capability from whatever the contract
    pins keeps the architecture pin in ``VERSIONS.env`` - it is deliberately
    not restated here - and still lets the wrapper reject a device that is not
    the pinned target.
    """
    match = _RE_CUDA_ARCH.match(cuda_arch)
    if match is None:
        raise P35Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    digits = match.group(1)
    if len(digits) < 2:
        raise P35Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    return f"{int(digits[:-1])}.{int(digits[-1])}"


# --- Upstream source identity ------------------------------------------------


def _git(args, cwd=None, safe_directory=None) -> str:
    """Run one read-only Git query and return its stripped stdout."""
    command = ["git"]
    if safe_directory is not None:
        # /opt/cutlass is a root-owned checkout inside the image while the
        # container runs as the invoking user, so each query carries its own
        # per-invocation safe.directory. Nothing is ever written there.
        command += ["-c", f"safe.directory={safe_directory}"]
    command += list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P35Error(f"git {' '.join(args)} could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P35Error(
            f"git {' '.join(args)} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def sha256_of_file(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise P35Error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def verify_upstream_sources(contract: dict) -> dict:
    """Prove both pinned upstream examples are byte-identical to their pins.

    Fails closed on a missing checkout, a wrong HEAD, any tracked or untracked
    modification, a symlinked or non-regular example file, a wrong Git blob
    SHA, or a wrong SHA-256 - for either file. The checkout is only ever
    queried, never written.
    """
    checkout = UPSTREAM_CHECKOUT_DIR
    if not checkout.is_dir():
        raise P35Error(f"pinned CUTLASS checkout {checkout} is missing")

    head = _git(["-C", str(checkout), "rev-parse", "HEAD"], safe_directory=str(checkout))
    if head != contract["CUTLASS_COMMIT"]:
        raise P35Error(
            f"{checkout} HEAD {head} != pinned CUTLASS_COMMIT {contract['CUTLASS_COMMIT']}"
        )

    dirty = _git(
        ["-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        safe_directory=str(checkout),
    )
    if dirty:
        raise P35Error(f"{checkout} has tracked or untracked modifications")

    sources = {}
    for name, source in sorted(UPSTREAM_SOURCES.items()):
        relative = contract[source["path_key"]]
        example = checkout / relative
        if example.is_symlink():
            raise P35Error(f"{example} is a symlink")
        if not example.is_file():
            raise P35Error(f"{example} is not a regular file")

        blob = _git(
            ["-C", str(checkout), "hash-object", "--", str(example)],
            safe_directory=str(checkout),
        )
        if blob != contract[source["blob_key"]]:
            raise P35Error(
                f"{relative} Git blob {blob} != pinned {contract[source['blob_key']]}"
            )

        sha256 = sha256_of_file(example)
        if sha256 != contract[source["sha256_key"]]:
            raise P35Error(
                f"{relative} SHA-256 {sha256} != pinned {contract[source['sha256_key']]}"
            )

        sources[name] = {
            "commit": head,
            "relative_path": relative,
            "path": example,
            "blob": blob,
            "sha256": sha256,
        }
    return sources


def load_upstream_module(example: Path, module_name: str):
    """Import a verified upstream example as a library, never as a script."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(example))
    if spec is None or spec.loader is None:
        raise P35Error(f"cannot build an import spec for {example}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - fail closed with the real cause
        sys.modules.pop(module_name, None)
        raise P35Error(f"cannot import the pinned upstream example {example}: {exc}") from exc
    return module


def load_upstream_modules(sources: dict) -> dict:
    """Import both verified examples and prove each provides what is needed."""
    modules = {}
    for name, source in sorted(UPSTREAM_SOURCES.items()):
        module = load_upstream_module(sources[name]["path"], source["module_name"])
        kernel_class = source["kernel_class"]
        if not hasattr(module, kernel_class):
            raise P35Error(
                f"the pinned {name} example does not provide {kernel_class}; P3.5 never "
                "substitutes another kernel class"
            )
        modules[name] = module

    factory_module = modules[OPERAND_FACTORY_SOURCE]
    if not hasattr(factory_module, OPERAND_FACTORY_NAME):
        raise P35Error(
            f"the pinned {OPERAND_FACTORY_SOURCE} example does not provide "
            f"{OPERAND_FACTORY_NAME}"
        )

    non_persistent_class = getattr(modules[SOURCE_NONPERSISTENT], UPSTREAM_CLASS_NONPERSISTENT)
    persistent_class = getattr(modules[SOURCE_PERSISTENT], UPSTREAM_CLASS_PERSISTENT)
    if non_persistent_class is persistent_class:
        raise P35Error(
            "the persistent and non-persistent kernel classes are the same object; "
            "the two schedulers cannot be distinguished"
        )
    return modules


def extract_upstream_matrix_calls(source: str) -> tuple:
    """Return the upstream tensor factory's ``matrix`` calls, in call order.

    Returns ``(calls, seeds)``. The upstream file is parsed, never imported
    here: this proof runs before any heavy import.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise P35Error(f"cannot parse the pinned upstream example: {exc}") from exc

    factory = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == OPERAND_FACTORY_NAME:
            factory = node
            break
    if factory is None:
        raise P35Error(
            f"the pinned upstream example no longer defines {OPERAND_FACTORY_NAME}()"
        )

    seeds = []
    calls = []
    for node in ast.walk(factory):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if name == "manual_seed" and len(node.args) == 1:
            if isinstance(node.args[0], ast.Constant):
                seeds.append(node.args[0].value)
        elif name == UPSTREAM_MATRIX_FACTORY:
            rendered = []
            for argument in node.args:
                if isinstance(argument, ast.Name):
                    rendered.append(argument.id)
                elif isinstance(argument, ast.Constant):
                    rendered.append(repr(argument.value))
                elif isinstance(argument, ast.Compare):
                    left = argument.left
                    rendered.append(left.id if isinstance(left, ast.Name) else "<expr>")
                else:
                    rendered.append("<expr>")
            calls.append((tuple(rendered), node.lineno))

    calls.sort(key=lambda item: item[1])
    return [{"args": args, "lineno": lineno} for args, lineno in calls], seeds


def verify_upstream_tensor_factory(example: Path) -> dict:
    """Prove the pinned factory still builds the operands assumed here.

    The upstream factory is called directly for the CuTe DSL candidates, and
    derives its cuBLASLt leading dimensions from the shapes and strides that
    same factory produces. Both therefore depend on the factory seeding once
    with the frozen seed and building A, then B, then C in that order, so a
    divergence has to fail the run rather than silently change the descriptors.
    """
    try:
        source = example.read_text(encoding="utf-8")
    except OSError as exc:
        raise P35Error(f"cannot read the pinned upstream example: {exc}") from exc

    calls, seeds = extract_upstream_matrix_calls(source)

    if FROZEN_SEED not in seeds:
        raise P35Error(
            f"the pinned upstream tensor factory does not seed with {FROZEN_SEED}; "
            "P3.5 cannot claim operand equivalence with P3.2, P3.3, and P3.4"
        )
    if len(calls) != len(UPSTREAM_MATRIX_CALLS):
        raise P35Error(
            f"the pinned upstream tensor factory makes {len(calls)} "
            f"{UPSTREAM_MATRIX_FACTORY}() call(s), P3.5 expects "
            f"{len(UPSTREAM_MATRIX_CALLS)}"
        )
    for index, (call, expected) in enumerate(zip(calls, UPSTREAM_MATRIX_CALLS)):
        args = call["args"]
        if len(args) < 5:
            raise P35Error(
                f"upstream {UPSTREAM_MATRIX_FACTORY}() call {index} has too few arguments: "
                f"{args}"
            )
        actual = (args[0], args[1], args[2], args[4])
        if actual != expected:
            raise P35Error(
                f"upstream {UPSTREAM_MATRIX_FACTORY}() call {index} is {actual}, P3.5 "
                f"expects {expected}; the operand construction has diverged"
            )
    return {"matrix_calls": [call["args"] for call in calls], "seeds": seeds}


# --- Environment and provenance ---------------------------------------------


def _query_nvidia_smi() -> dict:
    """Collect the allowlisted device fields for exactly one visible GPU."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P35Error(f"nvidia-smi could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P35Error(
            f"nvidia-smi failed with exit code {completed.returncode}; "
            "device provenance is ambiguous"
        )

    rows = [row for row in csv.reader(io.StringIO(completed.stdout)) if row]
    if len(rows) != 1:
        raise P35Error(f"nvidia-smi reported {len(rows)} GPUs; exactly 1 must be visible")
    fields = [value.strip() for value in rows[0]]
    if len(fields) != 3:
        raise P35Error("nvidia-smi returned a malformed device row")

    uuid, name, driver_version = fields
    if not _RE_GPU_UUID.match(uuid):
        raise P35Error(f"nvidia-smi returned a malformed GPU UUID: {uuid!r}")
    if not name or not _RE_SAFE_TEXT.match(name):
        raise P35Error("nvidia-smi returned a malformed GPU name")
    if not _RE_DOTTED_VERSION.match(driver_version):
        raise P35Error(f"nvidia-smi returned a malformed driver version: {driver_version!r}")
    return {"gpu_uuid": uuid, "gpu_name": name, "driver_version": driver_version}


def _query_nvcc_major_minor() -> str:
    """Read the installed CUDA toolkit's ``release X.Y`` from nvcc."""
    try:
        completed = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P35Error(f"nvcc could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P35Error("nvcc --version failed; the CUDA toolkit version is ambiguous")
    match = re.search(r"release ([0-9]+)\.([0-9]+)", completed.stdout)
    if match is None:
        raise P35Error("nvcc --version did not report a release version")
    return f"{match.group(1)}.{match.group(2)}"


def _repository_git_state(root: Path) -> dict:
    """Record this repository's commit and dirty state."""
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    if not _RE_HEX40.match(commit):
        raise P35Error(f"repository HEAD is malformed: {commit!r}")
    status = _git(["status", "--porcelain", "--untracked-files=all"], cwd=root)
    return {"git_commit": commit, "git_dirty": BOOL_TRUE if status else BOOL_FALSE}


def require_single_cuda_device(torch) -> None:
    """Require exactly one CUDA-visible GPU, used as logical device 0."""
    if not torch.cuda.is_available():
        raise P35Error("no CUDA device is available; P3.5 requires exactly one GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise P35Error(f"expected exactly 1 CUDA-visible GPU, saw {count}")
    torch.cuda.set_device(0)
    current = torch.cuda.current_device()
    if current != 0:
        raise P35Error(f"the selected CUDA device must be logical device 0, got {current}")


def require_ieee_fp32_matmul_api(torch):
    """Return the CUDA matmul backend, failing closed without the 2.10 API.

    This uses **exclusively** the PyTorch 2.10 ``fp32_precision`` API for CUDA
    matrix
    multiplication. The legacy ``allow_tf32`` property is never read and never
    written: in 2.10 the two are aliases of one setting, mixing them is
    unsupported, and the last write silently wins.
    ``torch.set_float32_matmul_precision()`` is likewise never combined with it.
    """
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul is None:
        raise P35Error(
            "this PyTorch does not expose torch.backends.cuda.matmul; the IEEE FP32 "
            "reference cannot be guaranteed"
        )
    if not hasattr(matmul, "fp32_precision"):
        raise P35Error(
            "this PyTorch does not support torch.backends.cuda.matmul.fp32_precision; "
            "P3.5 requires that API and never falls back to the legacy TF32 flag"
        )
    return matmul


@contextlib.contextmanager
def ieee_fp32_matmul(torch):
    """Guarantee IEEE FP32 CUDA matmul for the untimed correctness oracle."""
    matmul = require_ieee_fp32_matmul_api(torch)
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = FP32_PRECISION_IEEE
    except Exception as exc:  # noqa: BLE001 - any rejection is fail-closed
        raise P35Error(
            f"torch.backends.cuda.matmul.fp32_precision={FP32_PRECISION_IEEE!r} was "
            f"rejected: {exc}"
        ) from exc

    effective = matmul.fp32_precision
    if effective != FP32_PRECISION_IEEE:
        _restore_fp32_precision(matmul, previous)
        raise P35Error(
            f"torch.backends.cuda.matmul.fp32_precision read back as {effective!r}, "
            f"not {FP32_PRECISION_IEEE!r}; the FP32 reference cannot be trusted"
        )
    try:
        yield
    finally:
        _restore_fp32_precision(matmul, previous)


def _restore_fp32_precision(matmul, previous) -> None:
    """Restore the previous new-API setting, reporting a failure to stderr."""
    try:
        matmul.fp32_precision = previous
    except Exception as exc:  # noqa: BLE001 - never mask the original failure
        log(f"WARNING: could not restore fp32_precision to {previous!r}: {exc}")


def collect_provenance(contract: dict, torch, cutlass) -> dict:
    """Collect only the allowlisted provenance fields, failing closed.

    Nothing outside the allowlist is read or recorded: no host name, no user,
    no path, no environment dump. The result is identical for all twenty rows.
    """
    require_single_cuda_device(torch)

    device = _query_nvidia_smi()

    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = f"{major}.{minor}"
    if not _RE_COMPUTE_CAPABILITY.match(compute_capability):
        raise P35Error(f"malformed compute capability {compute_capability!r}")
    if compute_capability != contract["EXPECTED_COMPUTE_CAPABILITY"]:
        raise P35Error(
            f"device compute capability {compute_capability} does not match the pinned "
            f"{contract['CUDA_ARCH']} target ({contract['EXPECTED_COMPUTE_CAPABILITY']})"
        )

    nvcc_major_minor = _query_nvcc_major_minor()
    if nvcc_major_minor != contract["CUDA_MAJOR_MINOR"]:
        raise P35Error(
            f"installed CUDA toolkit {nvcc_major_minor} does not match the pinned "
            f"{contract['CUDA_VERSION']}"
        )

    torch_version = str(torch.__version__)
    if torch_version != contract["PYTORCH_VERSION"]:
        raise P35Error(f"torch {torch_version} != pinned {contract['PYTORCH_VERSION']}")
    torch_cuda_version = torch.version.cuda
    if torch_cuda_version != contract["PYTORCH_CUDA_VERSION"]:
        raise P35Error(
            f"torch CUDA {torch_cuda_version} != pinned {contract['PYTORCH_CUDA_VERSION']}"
        )

    cutedsl_version = str(cutlass.__version__)
    if cutedsl_version != contract["CUTEDSL_VERSION"]:
        raise P35Error(f"CuTe DSL {cutedsl_version} != pinned {contract['CUTEDSL_VERSION']}")

    git_state = _repository_git_state(repository_root())

    return {
        "gpu_name": device["gpu_name"],
        "gpu_uuid": device["gpu_uuid"],
        "compute_capability": compute_capability,
        "driver_version": device["driver_version"],
        "cuda_toolkit_version": contract["CUDA_VERSION"],
        "torch_cuda_version": torch_cuda_version,
        "cutedsl_version": cutedsl_version,
        "git_commit": git_state["git_commit"],
        "git_dirty": git_state["git_dirty"],
    }


# --- cuBLASLt bridge ---------------------------------------------------------


def _plan_info_type(ctypes):
    """Build the ctypes mirror of the bridge's ``P35PlanInfo`` struct.

    Every member is 8 bytes wide on both sides, so the layout is unambiguous;
    the loader additionally compares ``ctypes.sizeof`` against the size the
    compiled bridge reports and refuses to continue if they differ.
    """

    class P35PlanInfo(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_int64),
            ("cublaslt_version", ctypes.c_int64),
            ("shape_index", ctypes.c_int64),
            ("m", ctypes.c_int64),
            ("n", ctypes.c_int64),
            ("k", ctypes.c_int64),
            ("batch_count", ctypes.c_int64),
            ("lda", ctypes.c_int64),
            ("ldb", ctypes.c_int64),
            ("ldc", ctypes.c_int64),
            ("ldd", ctypes.c_int64),
            ("transa", ctypes.c_int64),
            ("transb", ctypes.c_int64),
            ("order_a", ctypes.c_int64),
            ("order_b", ctypes.c_int64),
            ("order_c", ctypes.c_int64),
            ("order_d", ctypes.c_int64),
            ("type_a", ctypes.c_int64),
            ("type_b", ctypes.c_int64),
            ("type_c", ctypes.c_int64),
            ("type_d", ctypes.c_int64),
            ("compute_type", ctypes.c_int64),
            ("scale_type", ctypes.c_int64),
            ("pointer_mode", ctypes.c_int64),
            ("epilogue", ctypes.c_int64),
            ("search_mode", ctypes.c_int64),
            ("workspace_limit_bytes", ctypes.c_int64),
            ("workspace_bytes", ctypes.c_int64),
            ("workspace_is_null", ctypes.c_int64),
            ("heuristic_requested", ctypes.c_int64),
            ("heuristic_returned", ctypes.c_int64),
            ("heuristic_index", ctypes.c_int64),
            ("alignment_a_bytes", ctypes.c_int64),
            ("alignment_b_bytes", ctypes.c_int64),
            ("alignment_c_bytes", ctypes.c_int64),
            ("alignment_d_bytes", ctypes.c_int64),
            ("algo_id", ctypes.c_int64),
            ("tile_id", ctypes.c_int64),
            ("stages_id", ctypes.c_int64),
            ("split_k", ctypes.c_int64),
            ("reduction_scheme", ctypes.c_int64),
            ("cta_swizzling", ctypes.c_int64),
            ("custom_option", ctypes.c_int64),
            ("inner_shape_id", ctypes.c_int64),
            ("cluster_shape_id", ctypes.c_int64),
            ("waves_count", ctypes.c_double),
            ("alpha", ctypes.c_double),
            ("beta", ctypes.c_double),
        ]

    return P35PlanInfo


class CublasLtBridge:
    """Thin, fail-closed ctypes front end for ``cublaslt_bridge_p35.cu``.

    The bridge owns every cuBLASLt object. This class owns nothing but the
    handle to it, translates a non-zero return code into a ``BridgeError``
    carrying the bridge's own diagnostic, and never silently retries. One plan
    exists at a time and is destroyed before the next shape begins.
    """

    def __init__(self, library_path: Path):
        import ctypes

        self._ctypes = ctypes
        if not library_path.is_file():
            raise BridgeError(
                f"the compiled P3.5 cuBLASLt bridge {library_path} is missing; the Make "
                f"targets build it from {BRIDGE_SOURCE_RELATIVE} immediately before this "
                "wrapper runs"
            )
        try:
            self._lib = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise BridgeError(f"cannot load the cuBLASLt bridge {library_path}: {exc}") from exc

        self.info_type = _plan_info_type(ctypes)

        self._lib.p35_bridge_abi_version.restype = ctypes.c_int
        self._lib.p35_bridge_abi_version.argtypes = []
        self._lib.p35_plan_info_size.restype = ctypes.c_size_t
        self._lib.p35_plan_info_size.argtypes = []
        self._lib.p35_last_error.restype = ctypes.c_char_p
        self._lib.p35_last_error.argtypes = []
        self._lib.p35_cublaslt_version.restype = ctypes.c_size_t
        self._lib.p35_cublaslt_version.argtypes = []
        self._lib.p35_shape_count.restype = ctypes.c_size_t
        self._lib.p35_shape_count.argtypes = []
        self._lib.p35_shape_at.restype = ctypes.c_int
        self._lib.p35_shape_at.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._lib.p35_plan_create.restype = ctypes.c_int
        self._lib.p35_plan_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(self.info_type),
        ]
        self._lib.p35_plan_execute.restype = ctypes.c_int
        self._lib.p35_plan_execute.argtypes = [ctypes.c_void_p]
        self._lib.p35_stream_synchronize.restype = ctypes.c_int
        self._lib.p35_stream_synchronize.argtypes = [ctypes.c_void_p]
        self._lib.p35_plan_destroy.restype = ctypes.c_int
        self._lib.p35_plan_destroy.argtypes = [ctypes.c_void_p]

        abi = int(self._lib.p35_bridge_abi_version())
        if abi != BRIDGE_ABI_VERSION:
            raise BridgeError(
                f"the compiled bridge reports ABI version {abi}, this wrapper expects "
                f"{BRIDGE_ABI_VERSION}"
            )
        reported_size = int(self._lib.p35_plan_info_size())
        mirror_size = ctypes.sizeof(self.info_type)
        if reported_size != mirror_size:
            raise BridgeError(
                f"the compiled bridge reports a {reported_size}-byte metadata struct, this "
                f"wrapper mirrors {mirror_size} bytes; the ABI does not match"
            )
        self._plan = None

    def _error(self) -> str:
        raw = self._lib.p35_last_error()
        if not raw:
            return "the bridge reported no diagnostic"
        return raw.decode("utf-8", errors="replace")

    def cublaslt_version(self) -> int:
        """The runtime cuBLASLt version, read from the library itself."""
        return int(self._lib.p35_cublaslt_version())

    def frozen_shapes(self) -> tuple:
        """Read the bridge's own shape allowlist, as (M,N,K) triples.

        The C translation unit and this module freeze the five shapes
        independently; reading the C side back is what turns "both sides agree"
        into a checked fact rather than an assumption.
        """
        ctypes = self._ctypes
        count = int(self._lib.p35_shape_count())
        if count <= 0 or count > 1024:
            raise BridgeError(f"the bridge reports {count} frozen shapes")
        shapes = []
        for index in range(count):
            m = ctypes.c_int64(0)
            n = ctypes.c_int64(0)
            k = ctypes.c_int64(0)
            status = self._lib.p35_shape_at(
                ctypes.c_size_t(index), ctypes.byref(m), ctypes.byref(n), ctypes.byref(k)
            )
            if status != 0:
                raise BridgeError(
                    f"the bridge could not report frozen shape {index}: {self._error()}"
                )
            shapes.append((int(m.value), int(n.value), int(k.value)))
        return tuple(shapes)

    def create_plan(self, m, n, k, a_ptr, b_ptr, c_ptr, d_ptr, stream):
        ctypes = self._ctypes
        if self._plan is not None:
            raise BridgeError(
                "a cuBLASLt plan already exists; P3.5 destroys each shape's plan before the "
                "next shape begins"
            )
        plan = ctypes.c_void_p(None)
        info = self.info_type()
        status = self._lib.p35_plan_create(
            ctypes.c_int64(int(m)),
            ctypes.c_int64(int(n)),
            ctypes.c_int64(int(k)),
            ctypes.c_void_p(a_ptr),
            ctypes.c_void_p(b_ptr),
            ctypes.c_void_p(c_ptr),
            ctypes.c_void_p(d_ptr),
            ctypes.c_void_p(stream),
            ctypes.byref(plan),
            ctypes.byref(info),
        )
        if status != 0:
            raise BridgeError(f"cuBLASLt plan creation failed: {self._error()}")
        if not plan:
            raise BridgeError("cuBLASLt plan creation reported success but returned no plan")
        self._plan = plan
        return info

    def execute(self) -> None:
        if self._plan is None:
            raise BridgeError("no cuBLASLt plan exists; nothing can be executed")
        if self._lib.p35_plan_execute(self._plan) != 0:
            raise BridgeError(f"cublasLtMatmul failed: {self._error()}")

    def synchronize(self) -> None:
        if self._plan is None:
            raise BridgeError("no cuBLASLt plan exists; nothing can be synchronized")
        if self._lib.p35_stream_synchronize(self._plan) != 0:
            raise BridgeError(f"stream synchronization failed: {self._error()}")

    def destroy(self) -> None:
        """Release this shape's plan, its workspace, and every descriptor."""
        if self._plan is None:
            return
        plan, self._plan = self._plan, None
        if self._lib.p35_plan_destroy(plan) != 0:
            raise BridgeError(f"releasing the cuBLASLt plan failed: {self._error()}")


def require_bridge_shape_allowlist(bridge) -> None:
    """Prove the C allowlist is exactly this module's five frozen shapes."""
    bridge_shapes = bridge.frozen_shapes()
    expected = tuple((m, n, k) for (m, n, k, _l) in FROZEN_SHAPES)
    if bridge_shapes != expected:
        raise BridgeError(
            f"the compiled bridge freezes the shapes {bridge_shapes}, this wrapper freezes "
            f"{expected}; P3.5 refuses to measure a shape set it did not specify"
        )


def validate_plan_info(info, mnkl, shape_index: int) -> dict:
    """Require the bridge's frozen contract to equal this wrapper's, exactly.

    The bridge freezes the layout, transpose modes, types, compute and scale
    types, pointer mode, epilogue, scalars, workspace limit, heuristic request
    count, search mode, and the shape allowlist in C; this wrapper freezes them
    in Python. Neither is derived from the other, so an accidental or deliberate
    change on one side is a disagreement here rather than a silently different
    measurement. The leading dimensions are derived from the shape on both
    sides and must agree.
    """
    m, n, k, l = mnkl
    expectations = (
        ("abi_version", BRIDGE_ABI_VERSION),
        ("shape_index", shape_index),
        ("m", m),
        ("n", n),
        ("k", k),
        ("batch_count", l),
        ("lda", k),
        ("ldb", k),
        ("ldc", n),
        ("ldd", n),
        ("transa", CUBLAS_ENUM_VALUES[FROZEN_TRANSA]),
        ("transb", CUBLAS_ENUM_VALUES[FROZEN_TRANSB]),
        ("order_a", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_b", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_c", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_d", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("type_a", CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE]),
        ("type_b", CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE]),
        ("type_c", CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE]),
        ("type_d", CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE]),
        ("compute_type", CUBLAS_ENUM_VALUES[FROZEN_COMPUTE_TYPE]),
        ("scale_type", CUBLAS_ENUM_VALUES[FROZEN_SCALE_TYPE]),
        ("pointer_mode", CUBLAS_ENUM_VALUES[FROZEN_POINTER_MODE]),
        ("epilogue", CUBLAS_ENUM_VALUES[FROZEN_EPILOGUE]),
        ("search_mode", CUBLAS_ENUM_VALUES[FROZEN_SEARCH_MODE]),
        ("workspace_limit_bytes", FROZEN_WORKSPACE_LIMIT_BYTES),
        ("heuristic_requested", FROZEN_HEURISTIC_REQUESTED),
    )
    for field, expected in expectations:
        actual = getattr(info, field)
        if int(actual) != int(expected):
            raise BridgeError(
                f"the compiled bridge reports {field}={actual}, this wrapper freezes "
                f"{expected}; P3.5 refuses to measure a configuration it did not specify"
            )

    if float(info.alpha) != FROZEN_ALPHA:
        raise BridgeError(f"the bridge reports alpha={info.alpha}, P3.5 freezes {FROZEN_ALPHA}")
    if float(info.beta) != FROZEN_BETA:
        raise BridgeError(f"the bridge reports beta={info.beta}, P3.5 freezes {FROZEN_BETA}")

    returned = int(info.heuristic_returned)
    index = int(info.heuristic_index)
    if not 1 <= returned <= FROZEN_HEURISTIC_REQUESTED:
        raise BridgeError(
            f"the heuristic returned {returned} result(s), which is outside "
            f"[1, {FROZEN_HEURISTIC_REQUESTED}]"
        )
    if not 0 <= index < returned:
        raise BridgeError(
            f"the selected heuristic index {index} is outside [0, {returned - 1}]"
        )

    workspace_bytes = int(info.workspace_bytes)
    if not 0 <= workspace_bytes <= FROZEN_WORKSPACE_LIMIT_BYTES:
        raise BridgeError(
            f"the selected algorithm needs {workspace_bytes} workspace bytes, outside "
            f"[0, {FROZEN_WORKSPACE_LIMIT_BYTES}]"
        )
    workspace_is_null = int(info.workspace_is_null)
    if workspace_is_null not in (0, 1):
        raise BridgeError(f"malformed workspace_is_null={workspace_is_null}")
    if (workspace_bytes == 0) != (workspace_is_null == 1):
        raise BridgeError(
            "the bridge must use a null workspace pointer exactly when the required "
            f"workspace is zero (bytes={workspace_bytes}, is_null={workspace_is_null})"
        )

    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes"):
        value = int(getattr(info, field))
        if value <= 0 or (value & (value - 1)) != 0:
            raise BridgeError(f"{field}={value} is not a positive power of two")

    waves = float(info.waves_count)
    if not math.isfinite(waves) or waves < 0.0:
        raise BridgeError(f"waves_count={waves!r} must be finite and non-negative")

    split_k = int(info.split_k)
    if split_k < 0:
        raise BridgeError(f"split_k={split_k} must be non-negative")

    for field in ("algo_id", "tile_id", "stages_id", "reduction_scheme", "cta_swizzling",
                  "custom_option", "inner_shape_id", "cluster_shape_id"):
        value = int(getattr(info, field))
        if value < 0:
            raise BridgeError(f"{field}={value} is negative; algorithm metadata is invalid")

    cublaslt_version = int(info.cublaslt_version)
    if cublaslt_version <= 0:
        raise BridgeError(f"cublasLtGetVersion() reported {cublaslt_version}")

    return {
        "order_a": FROZEN_ORDER,
        "order_b": FROZEN_ORDER,
        "order_c": FROZEN_ORDER,
        "order_d": FROZEN_ORDER,
        "transa": FROZEN_TRANSA,
        "transb": FROZEN_TRANSB,
        "lda": str(int(info.lda)),
        "ldb": str(int(info.ldb)),
        "ldc": str(int(info.ldc)),
        "ldd": str(int(info.ldd)),
        "compute_type": FROZEN_COMPUTE_TYPE,
        "scale_type": FROZEN_SCALE_TYPE,
        "pointer_mode": FROZEN_POINTER_MODE,
        "epilogue": FROZEN_EPILOGUE,
        "alpha": format_fixed(FROZEN_ALPHA, DECIMALS_SCALAR),
        "beta": format_fixed(FROZEN_BETA, DECIMALS_SCALAR),
        "search_mode": FROZEN_SEARCH_MODE,
        "workspace_limit_bytes": str(FROZEN_WORKSPACE_LIMIT_BYTES),
        "workspace_bytes": str(workspace_bytes),
        "alignment_a_bytes": str(int(info.alignment_a_bytes)),
        "alignment_b_bytes": str(int(info.alignment_b_bytes)),
        "alignment_c_bytes": str(int(info.alignment_c_bytes)),
        "alignment_d_bytes": str(int(info.alignment_d_bytes)),
        "heuristic_requested": str(FROZEN_HEURISTIC_REQUESTED),
        "heuristic_returned": str(returned),
        "heuristic_index": str(index),
        "algo_id": str(int(info.algo_id)),
        "tile_id": str(int(info.tile_id)),
        "stages_id": str(int(info.stages_id)),
        "split_k": str(split_k),
        "reduction_scheme": str(int(info.reduction_scheme)),
        "cta_swizzling": str(int(info.cta_swizzling)),
        "custom_option": str(int(info.custom_option)),
        "inner_shape_id": str(int(info.inner_shape_id)),
        "cluster_shape_id": str(int(info.cluster_shape_id)),
        "waves_count": format_fixed(waves, DECIMALS_WAVES),
        "cublaslt_version": str(cublaslt_version),
    }


# The comparison arithmetic is pure and GPU-independent.


def compute_flop_count(mnkl) -> int:
    """The exact FLOP count of one frozen GEMM: 2 x M x N x K.

    The factor of two counts one multiplication plus one addition per
    multiply-accumulate. Setup, compilation, the first launch, correctness, the
    output reset, and epilogue bookkeeping are all excluded by construction:
    this is a property of the problem, not of a measurement. The result is an
    exact Python integer - never a float - so no rounding can enter it.
    """
    m, n, k, l = mnkl
    for name, value in (("m", m), ("n", n), ("k", k), ("l", l)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ComparisonError(f"shape component {name}={value!r} must be a positive integer")
    if l != FROZEN_L:
        raise ComparisonError(f"P3.5 freezes L={FROZEN_L}, got {l}")
    return FLOPS_PER_MAC * m * n * k


def compute_tflops(flop_count: int, kernel_time_ms: float) -> float:
    """TFLOP/s from the exact FLOP count and the steady-state kernel time.

    ``flop_count / (kernel_time_ms * 1e9)``: dividing FLOP by milliseconds
    times 1e9 yields TFLOP/s directly, because 1 ms x 1e9 = 1e12 ns = the
    number of seconds that makes the quotient tera-scale. Only
    ``kernel_time_ms`` is used - never compilation, setup, or the first launch.
    """
    if isinstance(flop_count, bool) or not isinstance(flop_count, int) or flop_count <= 0:
        raise ComparisonError(f"flop_count={flop_count!r} must be a positive integer")
    time_ms = float(kernel_time_ms)
    if not math.isfinite(time_ms) or time_ms <= 0.0:
        raise ComparisonError(
            f"kernel_time_ms={kernel_time_ms!r} must be finite and strictly positive"
        )
    value = flop_count / (time_ms * 1e9)
    if not math.isfinite(value) or value <= 0.0:
        raise ComparisonError(f"the derived TFLOP/s {value!r} is not finite and positive")
    return value


def compute_shape_comparison(mnkl, kernel_times_ms) -> list:
    """Compute every comparison field for one shape's four candidates.

    ``kernel_times_ms`` is the full-precision steady-state time of each
    candidate, in the frozen candidate order. Returns one mapping per
    candidate, in that same order.

    The definitions are exactly:

        flop_count                   = 2 * M * N * K
        tflops                       = flop_count / (kernel_time_ms * 1e9)
        throughput_ratio_vs_cublaslt = candidate_tflops / cublaslt_tflops
        gap_to_cublaslt_pct          = 100 * (1 - throughput_ratio_vs_cublaslt)

    A positive gap means the candidate is slower than cuBLASLt, zero means
    equal, and a negative gap means the candidate is faster. Negative values
    are never clamped, and beating cuBLASLt is not a success requirement. The
    cuBLASLt row is the baseline and therefore carries an exact ratio of 1 and
    an exact gap of 0.

    Candidates are ranked by full-precision ``kernel_time_ms``, ascending, with
    an exact tie broken by the frozen candidate order. ``best_cutedsl_variant``
    is chosen among the three CuTe DSL candidates only, under the same rule,
    and is repeated identically on all four rows of the shape.

    No confidence interval, p-value, outlier removal, roofline efficiency,
    empirical-ceiling utilization, memory bandwidth, arithmetic-intensity
    classification, or causal interpretation is computed anywhere.
    """
    times = list(kernel_times_ms)
    if len(times) != FROZEN_CANDIDATE_COUNT:
        raise ComparisonError(
            f"a shape has exactly {FROZEN_CANDIDATE_COUNT} candidates, got {len(times)}"
        )
    values = []
    for index, raw in enumerate(times):
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ComparisonError(
                f"candidate {index} ({FROZEN_CANDIDATE_ORDER[index]}): kernel_time_ms="
                f"{raw!r} must be finite and strictly positive"
            )
        values.append(value)

    flop_count = compute_flop_count(mnkl)
    tflops = [compute_tflops(flop_count, value) for value in values]

    baseline_tflops = tflops[CUBLASLT_CANDIDATE_INDEX]
    if not math.isfinite(baseline_tflops) or baseline_tflops <= 0.0:
        raise ComparisonError("the cuBLASLt baseline TFLOP/s is not finite and positive")

    # Ranking: ascending steady-state time, exact ties broken by frozen order.
    order = sorted(range(FROZEN_CANDIDATE_COUNT), key=lambda index: (values[index], index))
    rank_within_shape = [0] * FROZEN_CANDIDATE_COUNT
    for position, index in enumerate(order):
        rank_within_shape[index] = position + 1

    # Best CuTe DSL variant: the same rule, restricted to the three CuTe rows.
    best_cutedsl_index = min(
        CUTEDSL_CANDIDATE_INDICES, key=lambda index: (values[index], index)
    )
    best_cutedsl_variant = FROZEN_CANDIDATE_ORDER[best_cutedsl_index]

    results = []
    for index in range(FROZEN_CANDIDATE_COUNT):
        if index == CUBLASLT_CANDIDATE_INDEX:
            # The baseline compares against itself: exactly 1 and exactly 0.
            ratio = 1.0
            gap = 0.0
        else:
            ratio = tflops[index] / baseline_tflops
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ComparisonError(
                    f"candidate {index}: the throughput ratio {ratio!r} is not finite and "
                    "positive"
                )
            gap = 100.0 * (1.0 - ratio)
            if not math.isfinite(gap):
                raise ComparisonError(f"candidate {index}: the gap {gap!r} is not finite")
        results.append(
            {
                "flop_count": flop_count,
                "tflops": tflops[index],
                "throughput_ratio_vs_cublaslt": ratio,
                "gap_to_cublaslt_pct": gap,
                "rank_within_shape": rank_within_shape[index],
                "best_cutedsl_variant": best_cutedsl_variant,
                "is_best_cutedsl": index == best_cutedsl_index,
            }
        )

    ranks = sorted(entry["rank_within_shape"] for entry in results)
    if ranks != list(range(1, FROZEN_CANDIDATE_COUNT + 1)):
        raise ComparisonError(f"the computed ranks {ranks} are not a permutation of 1..4")
    if sum(1 for entry in results if entry["is_best_cutedsl"]) != 1:
        raise ComparisonError("exactly one candidate must be the best CuTe DSL variant")
    if results[CUBLASLT_CANDIDATE_INDEX]["is_best_cutedsl"]:
        raise ComparisonError("the cuBLASLt baseline can never be the best CuTe DSL variant")
    return results


# --- CSV rows ----------------------------------------------------------------


def format_fixed(value, decimals: int) -> str:
    """Serialize a finite, non-negative real as a plain fixed-point decimal."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowContractError(f"{value!r} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise RowContractError(f"{value!r} is not finite; NaN and infinity are forbidden")
    if number < 0.0:
        raise RowContractError(f"{value!r} is negative")
    text = f"{number:.{decimals}f}"
    if text.startswith("-"):  # guards against a negative zero surviving rounding
        raise RowContractError(f"{value!r} serialized to a negative decimal")
    return text


def format_signed_fixed(value, decimals: int) -> str:
    """Serialize a finite, possibly negative real as a fixed-point decimal.

    ``gap_to_cublaslt_pct`` is the only signed field in the schema, and its sign
    carries the whole interpretation: positive means the candidate is slower
    than cuBLASLt, zero means equal, and negative means the candidate is faster.
    Negative values are therefore serialized as they are and never clamped.
    A value that rounds to zero from below is normalised to ``0.000000`` so
    that "equal" has exactly one spelling.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowContractError(f"{value!r} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise RowContractError(f"{value!r} is not finite; NaN and infinity are forbidden")
    text = f"{number:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text


def frozen_shape_at(shape_index: int):
    """Return the one frozen shape at an index, or fail."""
    if isinstance(shape_index, bool) or not isinstance(shape_index, int):
        raise RowContractError(f"shape_index {shape_index!r} is not an integer")
    if not 0 <= shape_index < FROZEN_SHAPE_COUNT:
        raise RowContractError(
            f"shape_index {shape_index} is outside [0, {FROZEN_SHAPE_COUNT - 1}]; P3.5 runs "
            f"exactly the {FROZEN_SHAPE_COUNT} frozen shapes and no other"
        )
    return FROZEN_SHAPES[shape_index]


def frozen_candidate_at(candidate_index: int) -> dict:
    """Return the one frozen candidate specification at an index, or fail."""
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise RowContractError(f"candidate_index {candidate_index!r} is not an integer")
    if not 0 <= candidate_index < FROZEN_CANDIDATE_COUNT:
        raise RowContractError(
            f"candidate_index {candidate_index} is outside "
            f"[0, {FROZEN_CANDIDATE_COUNT - 1}]; P3.5 runs exactly the "
            f"{FROZEN_CANDIDATE_COUNT} frozen candidates {FROZEN_CANDIDATE_ORDER}"
        )
    return FROZEN_CANDIDATES[candidate_index]


def _not_applicable_fields(method: str) -> tuple:
    """The fields that must carry the canonical marker for this method."""
    if method == METHOD_CUTEDSL:
        return CUBLASLT_ONLY_FIELDS
    if method == METHOD_CUBLASLT:
        return CUTEDSL_ONLY_FIELDS
    raise RowContractError(f"{method!r} is not one of the two frozen methods")


def build_row(
    shape_index: int,
    candidate_index: int,
    correctness: str,
    max_abs_error,
    max_rel_error,
    compile_time_ms,
    setup_time_ms,
    first_launch_ms,
    kernel_time_ms,
    warmup_iterations: int,
    iterations: int,
    max_active_clusters,
    comparison: dict,
    provenance: dict,
    operand_factory_sha256: str,
    upstream,
    plan,
) -> dict:
    """Build one frozen CSV row, refusing anything but a passed check.

    This is the only way a row is constructed, so a failed or skipped
    correctness check cannot produce an emittable row. The row's shape,
    method, variant, scheduler, tiler, cluster, and 2-CTA flag are taken from
    the frozen tables rather than from the caller, so a row can never describe
    a shape or configuration that was not one of the frozen ones.

    Exactly one of ``compile_time_ms`` (CuTe DSL) and ``setup_time_ms``
    (cuBLASLt) may be a number; the other must be ``None`` and is serialized as
    the canonical ``not_applicable``. The two are different concepts - JIT
    compilation versus vendor plan creation - and are never compared.
    """
    if correctness != CORRECTNESS_PASS:
        raise RowContractError(
            f"refusing to build a row with correctness={correctness!r}; "
            f"only {CORRECTNESS_PASS} may be emitted"
        )

    mnkl = frozen_shape_at(shape_index)
    spec = frozen_candidate_at(candidate_index)
    method = spec["method"]
    m, n, k, l = mnkl

    for name, value in (("first_launch_ms", first_launch_ms), ("kernel_time_ms", kernel_time_ms)):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise RowContractError(f"{name}={value!r} must be finite and strictly positive")

    if method == METHOD_CUTEDSL:
        if compile_time_ms is None:
            raise RowContractError(
                f"{spec['variant']}: a CuTe DSL candidate must report compile_time_ms"
            )
        if setup_time_ms is not None:
            raise RowContractError(
                f"{spec['variant']}: a CuTe DSL candidate has no cuBLASLt setup_time_ms, got "
                f"{setup_time_ms!r}"
            )
        compile_text = format_fixed(compile_time_ms, DECIMALS_TIMING)
        setup_text = NOT_APPLICABLE
        if float(compile_time_ms) <= 0.0:
            raise RowContractError("compile_time_ms must be strictly positive")
    else:
        if setup_time_ms is None:
            raise RowContractError(
                f"{spec['variant']}: a cuBLASLt candidate must report setup_time_ms"
            )
        if compile_time_ms is not None:
            raise RowContractError(
                f"{spec['variant']}: nothing is compiled at run time for cuBLASLt, so it has "
                f"no compile_time_ms, got {compile_time_ms!r}"
            )
        compile_text = NOT_APPLICABLE
        setup_text = format_fixed(setup_time_ms, DECIMALS_TIMING)
        if float(setup_time_ms) <= 0.0:
            raise RowContractError("setup_time_ms must be strictly positive")

    # max_active_clusters is a positive integer for a persistent CuTe DSL
    # candidate and the canonical marker for every other row.
    if spec["persistent"]:
        if isinstance(max_active_clusters, bool) or not isinstance(max_active_clusters, int):
            raise RowContractError(
                f"{spec['variant']}: max_active_clusters must be an integer, got "
                f"{max_active_clusters!r}"
            )
        if max_active_clusters <= 0:
            raise RowContractError(
                f"{spec['variant']}: max_active_clusters={max_active_clusters} must be positive"
            )
        max_active_clusters_text = str(max_active_clusters)
    else:
        if max_active_clusters is not None:
            raise RowContractError(
                f"{spec['variant']}: this candidate has no max_active_clusters, got "
                f"{max_active_clusters!r}"
            )
        max_active_clusters_text = NOT_APPLICABLE

    row = dict(CSV_FIXED_VALUES)
    row.update({field: NOT_APPLICABLE for field in _not_applicable_fields(method)})
    row.update(
        {
            "shape_index": str(shape_index + 1),
            "shape_id": shape_id(mnkl),
            "candidate_index": str(candidate_index + 1),
            "method": method,
            "variant": spec["variant"],
            "m": str(m),
            "n": str(n),
            "k": str(k),
            "l": str(l),
            "atol": format_fixed(FROZEN_ATOL, DECIMALS_TOLERANCE),
            "rtol": format_fixed(FROZEN_RTOL, DECIMALS_TOLERANCE),
            "max_abs_error": format_fixed(max_abs_error, DECIMALS_ERROR),
            "max_rel_error": format_fixed(max_rel_error, DECIMALS_ERROR),
            "compile_time_ms": compile_text,
            "setup_time_ms": setup_text,
            "first_launch_ms": format_fixed(first_launch_ms, DECIMALS_TIMING),
            "kernel_time_ms": format_fixed(kernel_time_ms, DECIMALS_TIMING),
            "warmup_iterations": str(int(warmup_iterations)),
            "iterations": str(int(iterations)),
            "cutlass_commit": upstream["commit"],
            "operand_factory_sha256": operand_factory_sha256,
        }
    )

    if method == METHOD_CUTEDSL:
        row.update(
            {
                "scheduler": spec["scheduler"],
                "mma_tiler_m": str(spec["mma_tiler_mn"][0]),
                "mma_tiler_n": str(spec["mma_tiler_mn"][1]),
                "cluster_m": str(spec["cluster_shape_mn"][0]),
                "cluster_n": str(spec["cluster_shape_mn"][1]),
                "use_2cta_instrs": BOOL_TRUE if spec["use_2cta_instrs"] else BOOL_FALSE,
                "use_tma_store": BOOL_TRUE if FROZEN_USE_TMA_STORE else BOOL_FALSE,
                "max_active_clusters": max_active_clusters_text,
                "upstream_kernel_file": upstream["relative_path"],
                "upstream_kernel_git_blob": upstream["blob"],
                "upstream_kernel_sha256": upstream["sha256"],
            }
        )
        if plan is not None:
            raise RowContractError(
                f"{spec['variant']}: a CuTe DSL candidate carries no cuBLASLt plan metadata"
            )
    else:
        if not isinstance(plan, dict):
            raise RowContractError(
                f"{spec['variant']}: the cuBLASLt candidate needs its plan metadata"
            )
        unexpected = sorted(set(plan) - set(CUBLASLT_ONLY_FIELDS))
        if unexpected:
            raise RowContractError(
                f"the cuBLASLt plan metadata carries unexpected field(s): "
                f"{', '.join(unexpected)}"
            )
        row.update(plan)
        # setup_time_ms belongs to the wrapper's timing discipline, not to the
        # plan metadata, so it is re-established after the update.
        row["setup_time_ms"] = setup_text

    row.update(_comparison_row_fields(comparison))
    row.update(provenance)
    validate_row(row)
    return row


def _comparison_row_fields(comparison: dict) -> dict:
    """Serialize one candidate's comparison fields deterministically."""
    if not isinstance(comparison, dict):
        raise RowContractError("the comparison result must be a mapping")
    required = {
        "flop_count", "tflops", "throughput_ratio_vs_cublaslt", "gap_to_cublaslt_pct",
        "rank_within_shape", "best_cutedsl_variant", "is_best_cutedsl",
    }
    missing = sorted(required - set(comparison))
    if missing:
        raise RowContractError(f"the comparison result is missing: {', '.join(missing)}")

    flop_count = comparison["flop_count"]
    if isinstance(flop_count, bool) or not isinstance(flop_count, int) or flop_count <= 0:
        raise RowContractError(f"flop_count={flop_count!r} must be a positive integer")

    rank = comparison["rank_within_shape"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise RowContractError(f"rank_within_shape={rank!r} must be an integer")
    if not 1 <= rank <= FROZEN_CANDIDATE_COUNT:
        raise RowContractError(
            f"rank_within_shape={rank} is outside [1, {FROZEN_CANDIDATE_COUNT}]"
        )

    best = comparison["best_cutedsl_variant"]
    if best not in tuple(FROZEN_CANDIDATE_ORDER[index] for index in CUTEDSL_CANDIDATE_INDICES):
        raise RowContractError(
            f"best_cutedsl_variant={best!r} is not one of the three CuTe DSL variants"
        )

    is_best = comparison["is_best_cutedsl"]
    if not isinstance(is_best, bool):
        raise RowContractError(f"is_best_cutedsl={is_best!r} must be a boolean")

    return {
        "flop_count": str(flop_count),
        "tflops": format_fixed(comparison["tflops"], DECIMALS_TFLOPS),
        "throughput_ratio_vs_cublaslt": format_fixed(
            comparison["throughput_ratio_vs_cublaslt"], DECIMALS_RATIO
        ),
        "gap_to_cublaslt_pct": format_signed_fixed(
            comparison["gap_to_cublaslt_pct"], DECIMALS_GAP
        ),
        "rank_within_shape": str(rank),
        "best_cutedsl_variant": best,
        "is_best_cutedsl": BOOL_TRUE if is_best else BOOL_FALSE,
    }


def _validate_bounded_count(field: str, text: str, maximum: int) -> None:
    value = int(text)
    if not MIN_ITERATIONS <= value <= maximum:
        raise RowContractError(f"{field}: {value} is outside [{MIN_ITERATIONS}, {maximum}]")


def _validate_decimal(field: str, text: str, decimals: int, strictly_positive: bool) -> None:
    if not re.fullmatch(rf"(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        raise RowContractError(
            f"{field}: {text!r} is not a fixed-point decimal with {decimals} fractional digits"
        )
    value = float(text)
    if not math.isfinite(value):
        raise RowContractError(f"{field}: {text!r} is not finite")
    if strictly_positive and value <= 0.0:
        raise RowContractError(f"{field}: {text!r} must be strictly positive")


def _validate_signed_decimal(field: str, text: str, decimals: int) -> None:
    """Accept a signed fixed-point decimal; a negative value is legitimate."""
    if not re.fullmatch(rf"-?(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        raise RowContractError(
            f"{field}: {text!r} is not a signed fixed-point decimal with {decimals} "
            "fractional digits"
        )
    value = float(text)
    if not math.isfinite(value):
        raise RowContractError(f"{field}: {text!r} is not finite")
    if text.startswith("-") and value == 0.0:
        raise RowContractError(
            f"{field}: {text!r} is a negative zero; an equal candidate is spelled "
            "unambiguously as zero"
        )


def validate_row(row) -> None:
    """Fail closed on any row that violates the frozen schema."""
    if not isinstance(row, dict):
        raise RowContractError("a CSV row must be a mapping")

    keys = set(row)
    expected = set(CSV_FIELDS)
    missing = sorted(expected - keys)
    if missing:
        raise RowContractError(f"missing field(s): {', '.join(missing)}")
    unknown = sorted(keys - expected)
    if unknown:
        raise RowContractError(f"unknown field(s): {', '.join(unknown)}")
    if len(CSV_FIELDS) != len(expected):
        raise RowContractError("the frozen schema contains a duplicate field name")

    for field in CSV_FIELDS:
        value = row[field]
        if not isinstance(value, str):
            raise RowContractError(f"{field}: value {value!r} is not a string")
        if value == "" or not _RE_SAFE_TEXT.match(value):
            raise RowContractError(
                f"{field}: value {value!r} is empty or contains control characters"
            )

    for field, fixed in CSV_FIXED_VALUES.items():
        if row[field] != fixed:
            raise RowContractError(f"{field}: {row[field]!r} != frozen {fixed!r}")

    # --- shape identity, from the frozen table only --------------------------
    if not _RE_POSITIVE_INT.match(row["shape_index"]):
        raise RowContractError(f"shape_index: {row['shape_index']!r} is not a positive integer")
    shape_index = int(row["shape_index"]) - 1
    mnkl = frozen_shape_at(shape_index)
    m, n, k, l = mnkl
    for field, expected_value in (
        ("m", str(m)), ("n", str(n)), ("k", str(k)), ("l", str(l)),
        ("shape_id", shape_id(mnkl)),
    ):
        if row[field] != expected_value:
            raise RowContractError(
                f"shape {row['shape_index']}: {field}={row[field]!r} != frozen "
                f"{expected_value!r}"
            )
    if not _RE_SHAPE_ID.match(row["shape_id"]):
        raise RowContractError(f"shape_id: {row['shape_id']!r} is malformed")

    # --- candidate identity, from the frozen table only ----------------------
    if not _RE_POSITIVE_INT.match(row["candidate_index"]):
        raise RowContractError(
            f"candidate_index: {row['candidate_index']!r} is not a positive integer"
        )
    candidate_index = int(row["candidate_index"]) - 1
    spec = frozen_candidate_at(candidate_index)
    if row["variant"] != spec["variant"]:
        raise RowContractError(
            f"candidate {row['candidate_index']}: variant={row['variant']!r} != frozen "
            f"{spec['variant']!r}"
        )
    if row["method"] != spec["method"]:
        raise RowContractError(
            f"{spec['variant']}: method={row['method']!r} != frozen {spec['method']!r}"
        )
    method = spec["method"]

    # --- method-specific applicability ---------------------------------------
    for field in _not_applicable_fields(method):
        if row[field] != NOT_APPLICABLE:
            raise RowContractError(
                f"{spec['variant']}: {field}={row[field]!r} must be {NOT_APPLICABLE!r} for "
                f"method={method}"
            )
    for field in (CUTEDSL_ONLY_FIELDS if method == METHOD_CUTEDSL else CUBLASLT_ONLY_FIELDS):
        if field == "max_active_clusters":
            continue  # handled below: not applicable for the non-persistent variant too
        if row[field] == NOT_APPLICABLE:
            raise RowContractError(
                f"{spec['variant']}: {field} must carry a real value for method={method}"
            )

    if method == METHOD_CUTEDSL:
        for field, expected_value in (
            ("scheduler", spec["scheduler"]),
            ("mma_tiler_m", str(spec["mma_tiler_mn"][0])),
            ("mma_tiler_n", str(spec["mma_tiler_mn"][1])),
            ("cluster_m", str(spec["cluster_shape_mn"][0])),
            ("cluster_n", str(spec["cluster_shape_mn"][1])),
            ("use_2cta_instrs", BOOL_TRUE if spec["use_2cta_instrs"] else BOOL_FALSE),
            ("use_tma_store", BOOL_TRUE if FROZEN_USE_TMA_STORE else BOOL_FALSE),
        ):
            if row[field] != expected_value:
                raise RowContractError(
                    f"{spec['variant']}: {field}={row[field]!r} != frozen {expected_value!r}"
                )
        if spec["persistent"]:
            if not _RE_POSITIVE_INT.match(row["max_active_clusters"]):
                raise RowContractError(
                    f"{spec['variant']}: max_active_clusters="
                    f"{row['max_active_clusters']!r} must be a positive decimal integer for a "
                    "persistent variant"
                )
        elif row["max_active_clusters"] != NOT_APPLICABLE:
            raise RowContractError(
                f"{spec['variant']}: max_active_clusters={row['max_active_clusters']!r} must "
                f"be {NOT_APPLICABLE!r} for the non-persistent variant"
            )
        _validate_decimal("compile_time_ms", row["compile_time_ms"], DECIMALS_TIMING, True)
        if not _RE_HEX40.match(row["upstream_kernel_git_blob"]):
            raise RowContractError("upstream_kernel_git_blob is not a 40-hex blob")
        if not _RE_HEX64.match(row["upstream_kernel_sha256"]):
            raise RowContractError("upstream_kernel_sha256 is not a 64-hex digest")
        if not is_relative_upstream_path(row["upstream_kernel_file"]):
            raise RowContractError(
                f"upstream_kernel_file: {row['upstream_kernel_file']!r} is not a relative "
                "upstream .py path"
            )
    else:
        _validate_cublaslt_fields(row, mnkl)
        _validate_decimal("setup_time_ms", row["setup_time_ms"], DECIMALS_TIMING, True)

    # --- shared numeric discipline -------------------------------------------
    for field in CSV_BOOL_FIELDS + ("is_best_cutedsl",):
        if row[field] not in (BOOL_TRUE, BOOL_FALSE):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical lowercase boolean"
            )
    if method == METHOD_CUBLASLT and row["is_best_cutedsl"] != BOOL_FALSE:
        raise RowContractError(
            "the cuBLASLt baseline row can never be marked as the best CuTe DSL variant"
        )

    for field in CSV_COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            raise RowContractError(f"{field}: {row[field]!r} is not a positive integer")
    _validate_bounded_count("warmup_iterations", row["warmup_iterations"], MAX_WARMUP_ITERATIONS)
    _validate_bounded_count("iterations", row["iterations"], MAX_ITERATIONS)

    for field in ("first_launch_ms", "kernel_time_ms"):
        _validate_decimal(field, row[field], DECIMALS_TIMING, strictly_positive=True)
    for field in CSV_ERROR_FIELDS:
        _validate_decimal(field, row[field], DECIMALS_ERROR, strictly_positive=False)
    for field in CSV_TOLERANCE_FIELDS:
        _validate_decimal(field, row[field], DECIMALS_TOLERANCE, strictly_positive=True)

    # --- the comparison fields ------------------------------------------------
    if not _RE_POSITIVE_INT.match(row["flop_count"]):
        raise RowContractError(f"flop_count: {row['flop_count']!r} is not a positive integer")
    expected_flop_count = compute_flop_count(mnkl)
    if int(row["flop_count"]) != expected_flop_count:
        raise RowContractError(
            f"flop_count: {row['flop_count']} != the exact 2*M*N*K value "
            f"{expected_flop_count} for shape {shape_id(mnkl)}"
        )
    _validate_decimal("tflops", row["tflops"], DECIMALS_TFLOPS, strictly_positive=True)
    _validate_decimal(
        "throughput_ratio_vs_cublaslt", row["throughput_ratio_vs_cublaslt"],
        DECIMALS_RATIO, strictly_positive=True,
    )
    _validate_signed_decimal("gap_to_cublaslt_pct", row["gap_to_cublaslt_pct"], DECIMALS_GAP)
    if not _RE_POSITIVE_INT.match(row["rank_within_shape"]):
        raise RowContractError(
            f"rank_within_shape: {row['rank_within_shape']!r} is not a positive integer"
        )
    if not 1 <= int(row["rank_within_shape"]) <= FROZEN_CANDIDATE_COUNT:
        raise RowContractError(
            f"rank_within_shape: {row['rank_within_shape']} is outside "
            f"[1, {FROZEN_CANDIDATE_COUNT}]"
        )
    if row["best_cutedsl_variant"] not in tuple(
        FROZEN_CANDIDATE_ORDER[index] for index in CUTEDSL_CANDIDATE_INDICES
    ):
        raise RowContractError(
            f"best_cutedsl_variant: {row['best_cutedsl_variant']!r} is not one of the three "
            "CuTe DSL variants"
        )
    if method == METHOD_CUBLASLT:
        # The baseline compares against itself: exactly 1 and exactly 0.
        if float(row["throughput_ratio_vs_cublaslt"]) != 1.0:
            raise RowContractError(
                f"the cuBLASLt baseline row must carry a throughput ratio of exactly 1, got "
                f"{row['throughput_ratio_vs_cublaslt']!r}"
            )
        if float(row["gap_to_cublaslt_pct"]) != 0.0:
            raise RowContractError(
                f"the cuBLASLt baseline row must carry a gap of exactly 0, got "
                f"{row['gap_to_cublaslt_pct']!r}"
            )

    # --- provenance -----------------------------------------------------------
    if not _RE_HEX40.match(row["cutlass_commit"]):
        raise RowContractError(
            f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit"
        )
    if not _RE_HEX40.match(row["git_commit"]):
        raise RowContractError(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["operand_factory_sha256"]):
        raise RowContractError("operand_factory_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        raise RowContractError(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        raise RowContractError(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in ("driver_version", "cuda_toolkit_version", "torch_cuda_version",
                  "cutedsl_version"):
        if not _RE_DOTTED_VERSION.match(row[field]):
            raise RowContractError(f"{field}: {row[field]!r} is not a dotted version")


def _validate_cublaslt_fields(row, mnkl) -> None:
    """Validate the descriptor, heuristic, and algorithm metadata of one row."""
    m, n, k, _l = mnkl
    for field, expected_value in (
        ("order_a", FROZEN_ORDER), ("order_b", FROZEN_ORDER),
        ("order_c", FROZEN_ORDER), ("order_d", FROZEN_ORDER),
        ("transa", FROZEN_TRANSA), ("transb", FROZEN_TRANSB),
        ("compute_type", FROZEN_COMPUTE_TYPE), ("scale_type", FROZEN_SCALE_TYPE),
        ("pointer_mode", FROZEN_POINTER_MODE), ("epilogue", FROZEN_EPILOGUE),
        ("search_mode", FROZEN_SEARCH_MODE),
        ("workspace_limit_bytes", str(FROZEN_WORKSPACE_LIMIT_BYTES)),
        ("heuristic_requested", str(FROZEN_HEURISTIC_REQUESTED)),
        # Derived from the shape on both sides of the ABI, never supplied.
        ("lda", str(k)), ("ldb", str(k)), ("ldc", str(n)), ("ldd", str(n)),
    ):
        if row[field] != expected_value:
            raise RowContractError(
                f"cuBLASLt row: {field}={row[field]!r} != frozen {expected_value!r}"
            )

    _validate_decimal("alpha", row["alpha"], DECIMALS_SCALAR, strictly_positive=True)
    _validate_decimal("beta", row["beta"], DECIMALS_SCALAR, strictly_positive=False)
    if float(row["alpha"]) != FROZEN_ALPHA:
        raise RowContractError(f"alpha: {row['alpha']!r} != frozen {FROZEN_ALPHA}")
    if float(row["beta"]) != FROZEN_BETA:
        raise RowContractError(f"beta: {row['beta']!r} != frozen {FROZEN_BETA}")
    _validate_decimal("waves_count", row["waves_count"], DECIMALS_WAVES, strictly_positive=False)

    for field in ("workspace_bytes", "heuristic_index", "algo_id", "tile_id", "stages_id",
                  "split_k", "reduction_scheme", "cta_swizzling", "custom_option",
                  "inner_shape_id", "cluster_shape_id", "cublaslt_version"):
        if not _RE_NONNEGATIVE_INT.match(row[field]):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical non-negative decimal integer"
            )
    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes", "heuristic_returned"):
        if not _RE_POSITIVE_INT.match(row[field]):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical positive decimal integer"
            )
    if int(row["cublaslt_version"]) <= 0:
        raise RowContractError("cublaslt_version must be positive")

    if int(row["workspace_bytes"]) > FROZEN_WORKSPACE_LIMIT_BYTES:
        raise RowContractError(
            f"workspace_bytes: {row['workspace_bytes']} exceeds the frozen limit "
            f"{FROZEN_WORKSPACE_LIMIT_BYTES}"
        )
    returned = int(row["heuristic_returned"])
    if returned > FROZEN_HEURISTIC_REQUESTED:
        raise RowContractError(
            f"heuristic_returned: {returned} exceeds the frozen request "
            f"{FROZEN_HEURISTIC_REQUESTED}"
        )
    if int(row["heuristic_index"]) >= returned:
        raise RowContractError(
            f"heuristic_index: {row['heuristic_index']} is not below heuristic_returned "
            f"{returned}"
        )
    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes"):
        value = int(row[field])
        if value & (value - 1):
            raise RowContractError(f"{field}: {value} is not a power of two")
    if int(row["m"]) != m:
        raise RowContractError("the cuBLASLt row's M does not match its frozen shape")


# Relative and absolute tolerances used only when a *serialized* decimal is
# re-checked against a quantity recomputed from other serialized decimals.
# Every decision is taken at full precision before serialization; these bounds
# exist purely to absorb the deterministic rounding of the fixed-point
# formats, and are far tighter than any real formula error could be.
COMPARISON_CHECK_RTOL = 1e-4
COMPARISON_CHECK_ATOL = 1e-6


def _close(actual: float, expected: float) -> bool:
    """Round-trip tolerance for a decimal re-derived from other decimals."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= COMPARISON_CHECK_ATOL + COMPARISON_CHECK_RTOL * abs(expected)


def validate_rows(rows) -> None:
    """Require exactly the twenty frozen measurements, in the frozen order.

    Shape-major, with the four candidates in the frozen order inside every
    shape. Every row is validated on its own, then the whole table is checked
    for completeness, provenance identity, and internal consistency of the
    comparison fields.
    """
    if not isinstance(rows, (list, tuple)):
        raise RowContractError("the P3.5 result must be a sequence of rows")
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RowContractError(
            f"P3.5 emits exactly {EXPECTED_ROW_COUNT} rows "
            f"({FROZEN_SHAPE_COUNT} shapes x {FROZEN_CANDIDATE_COUNT} candidates), "
            f"got {len(rows)}"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise RowContractError("every P3.5 result entry must be a mapping")
        validate_row(row)

    # Shape-major completeness and ordering.
    expected_keys = [
        (shape_index + 1, candidate_index + 1)
        for shape_index in range(FROZEN_SHAPE_COUNT)
        for candidate_index in range(FROZEN_CANDIDATE_COUNT)
    ]
    observed_keys = [(int(row["shape_index"]), int(row["candidate_index"])) for row in rows]
    if observed_keys != expected_keys:
        raise RowContractError(
            "the rows are not in shape-major frozen order (five shapes in order, four "
            f"candidates in order within each): got {observed_keys}"
        )
    observed_shape_ids = [row["shape_id"] for row in rows]
    for position, shape_index in enumerate(range(FROZEN_SHAPE_COUNT)):
        block = observed_shape_ids[
            shape_index * FROZEN_CANDIDATE_COUNT:(shape_index + 1) * FROZEN_CANDIDATE_COUNT
        ]
        if set(block) != {FROZEN_SHAPE_IDS[shape_index]}:
            raise RowContractError(
                f"shape block {position + 1} does not consistently describe "
                f"{FROZEN_SHAPE_IDS[shape_index]}: {block}"
            )
    if len(set(observed_shape_ids)) != FROZEN_SHAPE_COUNT:
        raise RowContractError(
            f"the twenty rows describe {len(set(observed_shape_ids))} distinct shapes, "
            f"expected {FROZEN_SHAPE_COUNT}"
        )

    # One run: identical provenance, seed, tolerances, and iteration counts.
    for field in (
        "gpu_name", "gpu_uuid", "compute_capability", "driver_version",
        "cuda_toolkit_version", "torch_cuda_version", "cutedsl_version", "cutlass_commit",
        "operand_factory_sha256", "git_commit", "git_dirty", "seed", "atol", "rtol",
        "warmup_iterations", "iterations", "schema_version", "experiment", "unit",
        "run_kind", "cache_mode", "publishable",
    ):
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise RowContractError(
                f"{field} differs between rows of one run: {sorted(values)}"
            )
    cublaslt_versions = {
        row["cublaslt_version"] for row in rows if row["method"] == METHOD_CUBLASLT
    }
    if len(cublaslt_versions) != 1:
        raise RowContractError(
            f"cublaslt_version differs between the five cuBLASLt rows: "
            f"{sorted(cublaslt_versions)}"
        )

    for shape_index in range(FROZEN_SHAPE_COUNT):
        block = rows[
            shape_index * FROZEN_CANDIDATE_COUNT:(shape_index + 1) * FROZEN_CANDIDATE_COUNT
        ]
        _validate_shape_block(shape_index, block)


def _validate_shape_block(shape_index: int, block) -> None:
    """Re-derive one shape's comparison fields from its four serialized rows."""
    mnkl = frozen_shape_at(shape_index)
    label = FROZEN_SHAPE_IDS[shape_index]

    baseline = block[CUBLASLT_CANDIDATE_INDEX]
    if baseline["method"] != METHOD_CUBLASLT:
        raise RowContractError(
            f"{label}: candidate {CUBLASLT_CANDIDATE_INDEX + 1} must be the cuBLASLt baseline"
        )

    flop_count = compute_flop_count(mnkl)
    times = [float(row["kernel_time_ms"]) for row in block]
    tflops = [float(row["tflops"]) for row in block]

    for index, row in enumerate(block):
        expected_tflops = compute_tflops(flop_count, times[index])
        if not _close(tflops[index], expected_tflops):
            raise RowContractError(
                f"{label}/{row['variant']}: tflops={row['tflops']} does not equal "
                f"flop_count/(kernel_time_ms*1e9) = {expected_tflops!r}"
            )

    baseline_tflops = tflops[CUBLASLT_CANDIDATE_INDEX]
    baseline_time = times[CUBLASLT_CANDIDATE_INDEX]
    for index, row in enumerate(block):
        ratio = float(row["throughput_ratio_vs_cublaslt"])
        gap = float(row["gap_to_cublaslt_pct"])
        if index == CUBLASLT_CANDIDATE_INDEX:
            if ratio != 1.0 or gap != 0.0:
                raise RowContractError(
                    f"{label}: the baseline row must carry ratio 1 and gap 0, got "
                    f"{row['throughput_ratio_vs_cublaslt']} / {row['gap_to_cublaslt_pct']}"
                )
            continue
        expected_ratio = tflops[index] / baseline_tflops
        if not _close(ratio, expected_ratio):
            raise RowContractError(
                f"{label}/{row['variant']}: throughput_ratio_vs_cublaslt={ratio!r} does not "
                f"equal candidate_tflops/cublaslt_tflops = {expected_ratio!r}"
            )
        # The equivalent time-based form must agree as well.
        expected_ratio_from_time = baseline_time / times[index]
        if not _close(ratio, expected_ratio_from_time):
            raise RowContractError(
                f"{label}/{row['variant']}: throughput_ratio_vs_cublaslt={ratio!r} does not "
                f"equal cublaslt_kernel_time_ms/candidate_kernel_time_ms = "
                f"{expected_ratio_from_time!r}"
            )
        expected_gap = 100.0 * (1.0 - ratio)
        if not _close(gap, expected_gap):
            raise RowContractError(
                f"{label}/{row['variant']}: gap_to_cublaslt_pct={gap!r} does not equal "
                f"100*(1 - throughput_ratio_vs_cublaslt) = {expected_gap!r}"
            )

    # Ranking: ascending full-precision time, exact ties by frozen order.
    order = sorted(range(FROZEN_CANDIDATE_COUNT), key=lambda index: (times[index], index))
    expected_rank = [0] * FROZEN_CANDIDATE_COUNT
    for position, index in enumerate(order):
        expected_rank[index] = position + 1
    observed_rank = [int(row["rank_within_shape"]) for row in block]
    if observed_rank != expected_rank:
        raise RowContractError(
            f"{label}: rank_within_shape {observed_rank} does not match the ascending "
            f"kernel_time_ms ranking {expected_rank} (ties broken by the frozen candidate "
            "order)"
        )
    if sorted(observed_rank) != list(range(1, FROZEN_CANDIDATE_COUNT + 1)):
        raise RowContractError(f"{label}: the ranks {observed_rank} are not 1..4")

    # Best CuTe DSL variant: same rule, restricted to the three CuTe rows.
    best_index = min(CUTEDSL_CANDIDATE_INDICES, key=lambda index: (times[index], index))
    expected_best = FROZEN_CANDIDATE_ORDER[best_index]
    declared = {row["best_cutedsl_variant"] for row in block}
    if declared != {expected_best}:
        raise RowContractError(
            f"{label}: best_cutedsl_variant is {sorted(declared)}, expected exactly "
            f"{expected_best!r} on all four rows"
        )
    flags = [row["is_best_cutedsl"] == BOOL_TRUE for row in block]
    if sum(flags) != 1:
        raise RowContractError(
            f"{label}: exactly one row must carry is_best_cutedsl=true, got {sum(flags)}"
        )
    if not flags[best_index]:
        raise RowContractError(
            f"{label}: is_best_cutedsl is set on {FROZEN_CANDIDATE_ORDER[flags.index(True)]!r}, "
            f"but the fastest CuTe DSL variant is {expected_best!r}"
        )
    for row in block:
        if int(row["flop_count"]) != flop_count:
            raise RowContractError(
                f"{label}/{row['variant']}: flop_count={row['flop_count']} != {flop_count}"
            )


def serialize_rows(rows) -> str:
    """Serialize all twenty validated rows with the csv module."""
    validate_rows(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


# --- stdout discipline -------------------------------------------------------


def _redirect_stdout_to_stderr() -> int:
    """Send everything written to descriptor 1 to stderr; return the real one.

    The JIT toolchain and the cuBLASLt library can write to descriptor 1 from
    native code, which would corrupt the twenty-one-line CSV contract.
    Redirecting at the descriptor level - rather than only rebinding
    ``sys.stdout`` - covers native writes too.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    return saved


def _emit_on_saved_stdout(saved_fd: int, text: str) -> None:
    """Write the CSV to the real stdout and close the saved descriptor."""
    with os.fdopen(saved_fd, "wb", closefd=True) as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()


# --- Operands and correctness ------------------------------------------------


def _describe(tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} strides={tuple(tensor.stride())} "
        f"dtype={tensor.dtype} device={tensor.device}"
    )


def require_operand_layout(tensor, name, shape, strides, dtype) -> None:
    """Fail closed unless a device operand has exactly the frozen layout.

    cuBLASLt is told the leading dimensions the frozen shape implies; if the
    tensor that actually backs a pointer had a different shape, stride, dtype,
    or device, the library would read the wrong memory and the measurement
    would be meaningless. Nothing here is corrected, transposed, or made
    contiguous: a mismatch is a hard failure.
    """
    if tensor.dtype is not dtype:
        raise P35Error(f"operand {name} has dtype {tensor.dtype}, expected {dtype}")
    if tensor.device.type != "cuda":
        raise P35Error(f"operand {name} is on {tensor.device}, expected a CUDA device")
    if tuple(tensor.shape) != tuple(shape):
        raise P35Error(f"operand {name} has {_describe(tensor)}, expected shape {tuple(shape)}")
    if tuple(tensor.stride()) != tuple(strides):
        raise P35Error(
            f"operand {name} has {_describe(tensor)}, expected strides {tuple(strides)}; "
            "P3.5 never silently transposes or re-lays out data"
        )
    if tensor.data_ptr() == 0:
        raise P35Error(f"operand {name} has a null device pointer")


def create_shape_operands(torch, cutlass, factory_module, mnkl) -> dict:
    """Build one shape's operands once, entirely outside every timer.

    The CuTe DSL candidates are handed exactly what the pinned non-persistent
    example's own ``create_tensors()`` produces - same factory, same seed
    ``1111``, same A/B/C creation order, same dtypes and strides - and A and B
    are never mutated afterwards.

    That factory keeps its device tensors for A and B private and returns only
    the host tensors and the device C buffer, so the cuBLASLt candidate cannot
    be handed a pointer into them. It is therefore given its own device copies,
    made from the very same immutable host tensors with the same
    ``empty_like``/``copy_`` pair, and the copies are proved
    byte-identical to those host tensors before anything runs. Byte-identity is
    additionally enforced end to end by the shared correctness oracle: all four
    candidates are validated against one reference computed from those same host
    tensors, so an operand that differed at all could not pass.
    """
    m, n, k, l = mnkl
    ab_dtype = getattr(cutlass, FROZEN_AB_DTYPE)
    c_dtype = getattr(cutlass, FROZEN_C_DTYPE)

    torch.manual_seed(FROZEN_SEED)
    (
        a_tensor,
        b_tensor,
        c_tensor,
        a_torch_cpu,
        b_torch_cpu,
        _c_torch_cpu,
        c_torch_gpu,
    ) = factory_module.create_tensors(
        l, m, n, k, FROZEN_A_MAJOR, FROZEN_B_MAJOR, FROZEN_C_MAJOR, ab_dtype, c_dtype
    )

    # The output buffer is what every candidate writes and what the
    # complete-result check reads, so its layout is part of the contract.
    require_operand_layout(
        c_torch_gpu, "C/D", (m, n, l), (n, 1, m * n), torch.float32
    )

    device_operands = {}
    for name, host in (("A", a_torch_cpu), ("B", b_torch_cpu)):
        device = torch.empty_like(host, device="cuda")
        device.copy_(host)
        torch.cuda.synchronize()
        if not bool(torch.equal(device.cpu(), host)):
            raise P35Error(
                f"the cuBLASLt device copy of operand {name} is not byte-identical to the "
                "host tensor the pinned upstream factory produced"
            )
        device_operands[name] = device

    require_operand_layout(
        device_operands["A"], "A", (m, k, l), (k, 1, m * k), torch.bfloat16
    )
    require_operand_layout(
        device_operands["B"], "B", (n, k, l), (k, 1, n * k), torch.bfloat16
    )

    return {
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
        "c_tensor": c_tensor,
        "a_torch_cpu": a_torch_cpu,
        "b_torch_cpu": b_torch_cpu,
        "c_torch_gpu": c_torch_gpu,
        "a_gpu": device_operands["A"],
        "b_gpu": device_operands["B"],
    }


def compute_reference(torch, a_torch_cpu, b_torch_cpu):
    """Compute the untimed IEEE-FP32 CUDA reference once per shape.

    The pinned PyTorch installation is used purely as a correctness
    reference. It is never
    timed, never reported as a competing method, and never compared against any
    candidate's timing. Because A and B are identical and immutable across all
    four candidates of a shape, one reference is correct for all of them.
    """
    with ieee_fp32_matmul(torch):
        reference = torch.einsum(
            "mkl,nkl->mnl",
            a_torch_cpu.to(device="cuda", dtype=torch.float32),
            b_torch_cpu.to(device="cuda", dtype=torch.float32),
        )
    if not bool(torch.isfinite(reference).all()):
        raise CorrectnessError("the FP32 reference contains non-finite values")
    return reference


def validate_result(torch, label, reference, c_torch_gpu):
    """Validate one candidate's complete result against the untimed reference.

    The output buffer was reset to NaN before this candidate ran, so any element
    the candidate failed to write is still non-finite here and is rejected.
    There is no fallback reference, no CPU reference, and no reduced-precision
    path.
    """
    result = c_torch_gpu.to(dtype=torch.float32)

    if tuple(result.shape) != tuple(reference.shape):
        raise CorrectnessError(
            f"{label}: result shape {tuple(result.shape)} != reference shape "
            f"{tuple(reference.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise CorrectnessError(
            f"{label}: the result contains non-finite values; the output buffer was reset to "
            "NaN before this candidate, so an element it did not write stays NaN"
        )

    difference = (result - reference).abs()
    tolerated = FROZEN_ATOL + FROZEN_RTOL * reference.abs()
    mismatches = int((difference > tolerated).sum().item())

    denominator = reference.abs().clamp_min(REL_ERROR_DENOMINATOR_FLOOR)
    max_abs_error = float(difference.max().item())
    max_rel_error = float((difference / denominator).max().item())

    if not math.isfinite(max_abs_error) or not math.isfinite(max_rel_error):
        raise CorrectnessError(f"{label}: the measured error is not finite")

    if mismatches:
        raise CorrectnessError(
            f"{label}: {mismatches} element(s) exceed atol={FROZEN_ATOL} "
            f"rtol={FROZEN_RTOL}; max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
        )
    return max_abs_error, max_rel_error


def _reset_output(torch, c_torch_gpu) -> None:
    """Reset the shared output buffer to a sentinel, outside every timer.

    NaN is used deliberately: any element a candidate fails to write stays
    non-finite and is rejected by the complete-result check, instead of
    surviving as a stale value from the previous candidate that would silently
    pass. This runs outside every timer and is followed by a synchronize.
    """
    c_torch_gpu.fill_(float("nan"))
    torch.cuda.synchronize()


# --- Per-candidate measurement -----------------------------------------------


def _max_active_clusters(cutlass, spec: dict) -> int:
    """Query the official pinned hardware helper for a persistent variant."""
    import cutlass.utils as utils

    cluster_m, cluster_n = spec["cluster_shape_mn"]
    cluster_size = cluster_m * cluster_n
    try:
        value = utils.HardwareInfo().get_max_active_clusters(cluster_size)
    except Exception as exc:  # noqa: BLE001 - fail closed with the real cause
        raise P35Error(
            f"{spec['variant']}: the official hardware helper could not report "
            f"max_active_clusters for cluster size {cluster_size}: {exc}"
        ) from exc

    if isinstance(value, bool) or not isinstance(value, int):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise P35Error(
                f"{spec['variant']}: max_active_clusters={value!r} is not a number"
            ) from exc
        if not math.isfinite(numeric) or numeric != int(numeric):
            raise P35Error(
                f"{spec['variant']}: max_active_clusters={value!r} is not a finite integer"
            )
        value = int(numeric)
    if value <= 0:
        raise P35Error(
            f"{spec['variant']}: max_active_clusters={value} must be a positive integer"
        )
    return value


def _build_kernel(module, spec: dict, cutlass):
    """Instantiate the frozen kernel object for one CuTe DSL candidate."""
    kernel_class = getattr(module, spec["upstream_class"])
    acc_dtype = getattr(cutlass, FROZEN_ACC_DTYPE)
    return kernel_class(
        acc_dtype,
        spec["use_2cta_instrs"],
        spec["mma_tiler_mn"],
        spec["cluster_shape_mn"],
        FROZEN_USE_TMA_STORE,
    )


def _can_implement(gemm, spec: dict, mnkl, a_tensor, b_tensor, c_tensor) -> None:
    """Run the official ``can_implement()`` check for this candidate's class.

    The two upstream classes deliberately expose different signatures:
    ``DenseGemmKernel.can_implement(a, b, c)`` takes the tensors, while
    ``PersistentDenseGemmKernel.can_implement(mnkl, a_dtype, b_dtype, c_dtype,
    a_major, b_major, c_major)`` takes the problem description. Each is called
    in its own official form; there is never a fallback to another
    configuration when a check fails, for any shape.
    """
    if spec["persistent"]:
        supported = gemm.can_implement(
            tuple(mnkl),
            a_tensor.element_type,
            b_tensor.element_type,
            c_tensor.element_type,
            FROZEN_A_MAJOR,
            FROZEN_B_MAJOR,
            FROZEN_C_MAJOR,
        )
    else:
        supported = gemm.can_implement(a_tensor, b_tensor, c_tensor)
    if not supported:
        raise P35Error(
            f"{shape_id(mnkl)}/{spec['variant']}: the pinned {spec['upstream_class']} cannot "
            f"implement the frozen configuration (tiler {spec['mma_tiler_mn']}, cluster "
            f"{spec['cluster_shape_mn']}, use_2cta_instrs={spec['use_2cta_instrs']}); "
            "P3.5 never falls back to another configuration"
        )


def _launch(compiled_gemm, spec: dict, a_tensor, b_tensor, c_tensor, stream) -> None:
    """Launch a compiled CuTe kernel with its dynamic-only runtime signature.

    ``cute.compile`` bakes every ``cutlass.Constexpr`` parameter in at compile
    time and drops it from the compiled callable, which therefore takes only
    the dynamic arguments. Both pinned examples demonstrate exactly this. A
    TypeError here means that contract changed, which is a hard failure rather
    than something to work around.
    """
    try:
        compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
    except TypeError as exc:
        raise P35Error(
            f"{spec['variant']}: the compiled kernel rejected the dynamic-only launch "
            f"signature (a, b, c, stream): {exc}. The pinned CuTe DSL is expected to bake "
            "every cutlass.Constexpr argument in at compile time; P3.5 does not guess "
            "another signature"
        ) from exc


def _measure_cutedsl_candidate(
    spec, mnkl, modules, cutlass, cute, torch, operands, reference,
    torch_stream, cute_stream, warmup_iterations, iterations,
) -> dict:
    """Run the frozen sequence for one CuTe DSL candidate of one shape."""
    label = f"{shape_id(mnkl)}/{spec['variant']}"
    module = modules[spec["source"]]

    gemm = _build_kernel(module, spec, cutlass)
    _can_implement(
        gemm, spec, mnkl, operands["a_tensor"], operands["b_tensor"], operands["c_tensor"]
    )
    log(f"{label}: can_implement OK")

    if spec["persistent"]:
        max_active_clusters = _max_active_clusters(cutlass, spec)
        log(f"{label}: max_active_clusters={max_active_clusters} (official helper)")
        compile_args = (
            operands["a_tensor"], operands["b_tensor"], operands["c_tensor"],
            max_active_clusters, cute_stream,
        )
    else:
        max_active_clusters = None
        compile_args = (
            operands["a_tensor"], operands["b_tensor"], operands["c_tensor"], cute_stream
        )

    # Reset and synchronize, outside every timer.
    _reset_output(torch, operands["c_torch_gpu"])

    # Compilation only. This is compile_time_ms and never setup_time_ms.
    log(f"{label}: compiling (JIT)")
    torch.cuda.synchronize()
    compile_start = time.perf_counter_ns()
    compiled_gemm = cute.compile(gemm, *compile_args)
    torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter_ns() - compile_start) / 1e6

    # First launch; its output is the tensor that gets validated.
    log(f"{label}: first launch (also the correctness-validated launch)")
    torch.cuda.synchronize()
    first_launch_start = time.perf_counter_ns()
    _launch(
        compiled_gemm, spec, operands["a_tensor"], operands["b_tensor"],
        operands["c_tensor"], cute_stream,
    )
    torch.cuda.synchronize()
    first_launch_ms = (time.perf_counter_ns() - first_launch_start) / 1e6

    max_abs_error, max_rel_error = validate_result(
        torch, label, reference, operands["c_torch_gpu"]
    )
    log(
        f"{label}: correctness {CORRECTNESS_PASS} "
        f"(max_abs_error={max_abs_error!r} max_rel_error={max_rel_error!r})"
    )

    # Warm-up, only after this candidate passed correctness.
    log(f"{label}: warm-up {warmup_iterations} launch(es)")
    for _ in range(warmup_iterations):
        _launch(
            compiled_gemm, spec, operands["a_tensor"], operands["b_tensor"],
            operands["c_tensor"], cute_stream,
        )
    torch.cuda.synchronize()

    # Steady state on the kernel's own stream.
    log(f"{label}: steady state {iterations} measured launch(es)")
    kernel_time_ms = _measure_steady_state(
        torch,
        torch_stream,
        iterations,
        label,
        lambda: _launch(
            compiled_gemm, spec, operands["a_tensor"], operands["b_tensor"],
            operands["c_tensor"], cute_stream,
        ),
    )

    return {
        "compile_time_ms": compile_time_ms,
        "setup_time_ms": None,
        "first_launch_ms": first_launch_ms,
        "kernel_time_ms": kernel_time_ms,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "max_active_clusters": max_active_clusters,
        "plan": None,
    }


def _measure_cublaslt_candidate(
    spec, shape_index, mnkl, torch, operands, reference, torch_stream,
    warmup_iterations, iterations,
) -> dict:
    """Run the frozen sequence for the cuBLASLt candidate of one shape."""
    label = f"{shape_id(mnkl)}/{spec['variant']}"

    bridge = CublasLtBridge(BRIDGE_LIBRARY_PATH)
    require_bridge_shape_allowlist(bridge)
    log(f"{label}: cuBLASLt runtime version {bridge.cublaslt_version()}")

    # C and D are the same FP32 M x N buffer: beta is exactly 0, so C is never
    # read, and the descriptor contract gives C and D identical layouts.
    c_ptr = operands["c_torch_gpu"].data_ptr()
    m, n, k, _l = mnkl

    try:
        # Reset and synchronize, outside every timer.
        _reset_output(torch, operands["c_torch_gpu"])

        # Plan creation only: descriptors, layouts, preference, the vendor
        # heuristic, algorithm validation, and the workspace. No matmul runs
        # inside this timer and nothing is compiled - which is exactly why this
        # field is setup_time_ms and never compile_time_ms. The two are
        # different concepts and are never compared against each other.
        log(f"{label}: creating the cuBLASLt plan (descriptors, heuristic, workspace)")
        torch.cuda.synchronize()
        setup_start = time.perf_counter_ns()
        info = bridge.create_plan(
            m, n, k,
            operands["a_gpu"].data_ptr(), operands["b_gpu"].data_ptr(),
            c_ptr, c_ptr, torch_stream.cuda_stream,
        )
        setup_time_ms = (time.perf_counter_ns() - setup_start) / 1e6

        plan_fields = validate_plan_info(info, mnkl, shape_index)
        log(
            f"{label}: heuristic requested {FROZEN_HEURISTIC_REQUESTED}, returned "
            f"{plan_fields['heuristic_returned']}, selected index "
            f"{plan_fields['heuristic_index']} (algo_id {plan_fields['algo_id']}, "
            f"tile {plan_fields['tile_id']}, stages {plan_fields['stages_id']}, "
            f"split_k {plan_fields['split_k']}, workspace "
            f"{plan_fields['workspace_bytes']} B)"
        )

        torch.cuda.synchronize()
        log(f"{label}: first cublasLtMatmul launch (also the correctness-validated launch)")
        first_launch_start = time.perf_counter_ns()
        bridge.execute()
        torch.cuda.synchronize()
        first_launch_ms = (time.perf_counter_ns() - first_launch_start) / 1e6

        max_abs_error, max_rel_error = validate_result(
            torch, label, reference, operands["c_torch_gpu"]
        )
        log(
            f"{label}: correctness {CORRECTNESS_PASS} "
            f"(max_abs_error={max_abs_error!r} max_rel_error={max_rel_error!r})"
        )

        log(f"{label}: warm-up {warmup_iterations} launch(es)")
        for _ in range(warmup_iterations):
            bridge.execute()
        torch.cuda.synchronize()

        log(f"{label}: steady state {iterations} measured launch(es)")
        kernel_time_ms = _measure_steady_state(
            torch, torch_stream, iterations, label, bridge.execute
        )
    finally:
        # The plan, its workspace, and every descriptor are shape-owned and are
        # released before the next shape begins. A cleanup error invalidates an
        # otherwise successful candidate, but never masks its primary failure.
        _cleanup_preserving_primary(bridge.destroy, f"releasing the cuBLASLt plan for {label}")

    return {
        "compile_time_ms": None,
        "setup_time_ms": setup_time_ms,
        "first_launch_ms": first_launch_ms,
        "kernel_time_ms": kernel_time_ms,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "max_active_clusters": None,
        "plan": plan_fields,
    }


def _measure_steady_state(torch, torch_stream, iterations, label, launch) -> float:
    """CUDA-event steady state on the candidate's own stream, per iteration."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch_stream)
    for _ in range(iterations):
        launch()
    end_event.record(torch_stream)
    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)
    if not math.isfinite(total_ms) or total_ms <= 0.0:
        raise P35Error(
            f"{label}: CUDA-event elapsed time {total_ms!r} is not finite and positive"
        )
    kernel_time_ms = total_ms / iterations
    if not math.isfinite(kernel_time_ms) or kernel_time_ms <= 0.0:
        raise P35Error(
            f"{label}: kernel_time_ms={kernel_time_ms!r} is not finite and strictly positive"
        )
    return kernel_time_ms


# --- Orchestration -----------------------------------------------------------


def execute_measurement(warmup_iterations: int, iterations: int) -> str:
    """Run all five shapes and all four candidates and return the CSV text.

    Returning the text rather than writing it keeps the single emission point
    in ``main`` and makes it structurally impossible to emit a partial table: a
    failure at any shape or candidate propagates before anything reaches
    stdout, including rows already completed.
    """
    contract = load_pinned_contract()

    # (1) Both pinned upstream identities are proved before anything heavy is
    # imported and certainly before either module is loaded.
    sources = verify_upstream_sources(contract)
    for name in sorted(sources):
        log(
            f"upstream verified ({name}): {sources[name]['relative_path']} "
            f"blob {sources[name]['blob']} sha256 {sources[name]['sha256']}"
        )
    factory_source = sources[OPERAND_FACTORY_SOURCE]
    factory = verify_upstream_tensor_factory(factory_source["path"])
    log(
        f"operand construction: the pinned factory still seeds {FROZEN_SEED} and builds "
        f"{len(factory['matrix_calls'])} matrices in the order P3.5 depends on"
    )

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch

    # (2) Environment and provenance, before any tensor exists.
    log("collecting environment and provenance")
    require_ieee_fp32_matmul_api(torch)
    provenance = collect_provenance(contract, torch, cutlass)

    revalidated = verify_upstream_sources(contract)
    if revalidated != sources:
        raise P35Error("a pinned upstream source changed during provenance collection")
    log(
        f"device: {provenance['gpu_name']} uuid={provenance['gpu_uuid']} "
        f"cc={provenance['compute_capability']} driver={provenance['driver_version']}"
    )

    modules = load_upstream_modules(sources)
    factory_module = modules[OPERAND_FACTORY_SOURCE]
    operand_factory_sha256 = factory_source["sha256"]

    torch_stream = torch.cuda.current_stream()
    cute_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    rows = []
    for shape_index, mnkl in enumerate(FROZEN_SHAPES):
        rows.extend(
            _measure_shape(
                shape_index=shape_index,
                mnkl=mnkl,
                modules=modules,
                factory_module=factory_module,
                sources=sources,
                cutlass=cutlass,
                cute=cute,
                torch=torch,
                torch_stream=torch_stream,
                cute_stream=cute_stream,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                provenance=provenance,
                operand_factory_sha256=operand_factory_sha256,
            )
        )

    # Only a fully completed sweep of all five shapes and all four candidates
    # reaches this line.
    return serialize_rows(rows)


def _measure_shape(
    shape_index, mnkl, modules, factory_module, sources, cutlass, cute, torch,
    torch_stream, cute_stream, warmup_iterations, iterations, provenance,
    operand_factory_sha256,
) -> list:
    """Measure all four candidates of one shape and return its four rows."""
    label = shape_id(mnkl)
    log(f"=== shape {shape_index + 1}/{FROZEN_SHAPE_COUNT}: (M,N,K,L)={tuple(mnkl)} ===")

    operands = None
    reference = None
    try:
        # One shared operand set per shape, entirely outside every timer.
        log(f"{label}: allocating the shared operands once (outside every timer)")
        operands = create_shape_operands(torch, cutlass, factory_module, mnkl)

        # One untimed reference per shape, reused by all four candidates.
        log(f"{label}: computing the untimed IEEE-FP32 reference once (outside every timer)")
        reference = compute_reference(
            torch, operands["a_torch_cpu"], operands["b_torch_cpu"]
        )

        measurements = []
        for candidate_index, spec in enumerate(FROZEN_CANDIDATES):
            log(
                f"--- {label} candidate {candidate_index + 1}/{FROZEN_CANDIDATE_COUNT}: "
                f"{spec['method']}/{spec['variant']} ---"
            )
            if spec["method"] == METHOD_CUTEDSL:
                measurements.append(
                    _measure_cutedsl_candidate(
                        spec=spec, mnkl=mnkl, modules=modules, cutlass=cutlass, cute=cute,
                        torch=torch, operands=operands, reference=reference,
                        torch_stream=torch_stream, cute_stream=cute_stream,
                        warmup_iterations=warmup_iterations, iterations=iterations,
                    )
                )
            else:
                measurements.append(
                    _measure_cublaslt_candidate(
                        spec=spec, shape_index=shape_index, mnkl=mnkl, torch=torch,
                        operands=operands, reference=reference, torch_stream=torch_stream,
                        warmup_iterations=warmup_iterations, iterations=iterations,
                    )
                )

        # Every candidate of this shape has now passed correctness and produced
        # a steady-state time, so the comparison can be computed at full
        # precision and serialized deterministically.
        comparison = compute_shape_comparison(
            mnkl, [entry["kernel_time_ms"] for entry in measurements]
        )
        log(
            f"{label}: best CuTe DSL variant {comparison[0]['best_cutedsl_variant']}; "
            f"ranks {[entry['rank_within_shape'] for entry in comparison]} "
            "(descriptive only, non-publishable)"
        )

        shape_rows = []
        for candidate_index, (spec, measurement) in enumerate(
            zip(FROZEN_CANDIDATES, measurements)
        ):
            upstream = (
                sources[spec["source"]] if spec["method"] == METHOD_CUTEDSL
                else sources[OPERAND_FACTORY_SOURCE]
            )
            shape_rows.append(
                build_row(
                    shape_index=shape_index,
                    candidate_index=candidate_index,
                    correctness=CORRECTNESS_PASS,
                    max_abs_error=measurement["max_abs_error"],
                    max_rel_error=measurement["max_rel_error"],
                    compile_time_ms=measurement["compile_time_ms"],
                    setup_time_ms=measurement["setup_time_ms"],
                    first_launch_ms=measurement["first_launch_ms"],
                    kernel_time_ms=measurement["kernel_time_ms"],
                    warmup_iterations=warmup_iterations,
                    iterations=iterations,
                    max_active_clusters=measurement["max_active_clusters"],
                    comparison=comparison[candidate_index],
                    provenance=provenance,
                    operand_factory_sha256=operand_factory_sha256,
                    upstream=upstream,
                    plan=measurement["plan"],
                )
            )
        return shape_rows
    finally:
        # Shape-owned tensors are released before the next shape begins, so the
        # five shapes never hold their operands simultaneously. Synchronization
        # or allocator-cleanup failure invalidates an otherwise successful shape.
        del reference
        del operands

        def release_shape_memory():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        _cleanup_preserving_primary(
            release_shape_memory, f"releasing shape {label} device memory"
        )


# --- Command line ------------------------------------------------------------


def bounded_int(minimum: int, maximum: int):
    """argparse type for a positive, explicitly bounded iteration count."""

    def parse(text: str) -> int:
        stripped = text.strip()
        if not re.fullmatch(r"[0-9]+", stripped):
            raise argparse.ArgumentTypeError(f"{text!r} is not a non-negative integer")
        value = int(stripped)
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{value} is outside the permitted range [{minimum}, {maximum}]"
            )
        return value

    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the whole command line.

    The frozen scientific contract is deliberately unreachable from here: there
    is no shape, method, variant, dtype, layout, scheduler, tile, cluster, seed,
    tolerance, workspace, algorithm, source-path, publication, correctness-skip,
    partial-run, or output-file control, and no way to run fewer than all five
    shapes and all four candidates.
    """
    parser = argparse.ArgumentParser(
        prog="gemm_comparison.py",
        description=(
            "Five shapes and comparison. Executes exactly four frozen candidates - "
            "nonpersistent_1cta, persistent_1cta, persistent_2cta, and cuBLASLt "
            "heuristic_first_supported - on each of the five frozen final shapes, on "
            "identical operands and against one untimed FP32 oracle per shape, and emits "
            "twenty non-publishable CSV rows with descriptive comparison fields. Not an "
            "experimental campaign, not a statistical treatment, and not a performance claim."
        ),
        epilog=(
            "Correctness is mandatory and always runs before any warm-up or steady-state "
            "timing, per candidate. All twenty-one output lines are emitted only after all "
            "five shapes and all four candidates pass. Every row is publishable=false: the "
            "comparison fields are arithmetic, not a conclusion, and no pilot, campaign, or "
            "downstream interpretation has been performed."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_WARMUP_ITERATIONS),
        default=DEFAULT_WARMUP_ITERATIONS,
        metavar="N",
        help=(
            f"untimed launches before the measured ones, per candidate "
            f"[{MIN_ITERATIONS}..{MAX_WARMUP_ITERATIONS}], default {DEFAULT_WARMUP_ITERATIONS}"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_ITERATIONS),
        default=DEFAULT_ITERATIONS,
        metavar="N",
        help=(
            f"measured launches for kernel_time_ms, per candidate "
            f"[{MIN_ITERATIONS}..{MAX_ITERATIONS}], default {DEFAULT_ITERATIONS}"
        ),
    )
    return parser


# --- Entry point -------------------------------------------------------------


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Descriptor 1 becomes stderr for the whole measurement; the real stdout is
    # restored only to emit the twenty-one CSV lines, and only after every one
    # of the twenty measurements has passed validation.
    saved_stdout_fd = _redirect_stdout_to_stderr()
    try:
        csv_text = execute_measurement(args.warmup_iterations, args.iterations)
    except CorrectnessError as exc:
        os.close(saved_stdout_fd)
        log(f"CORRECTNESS FAILED: {exc}")
        log(
            "no CSV header and no CSV row are emitted, including for shapes and candidates "
            "that already passed; that candidate ran no warm-up and no steady-state timing"
        )
        log("this run is not a completed comparison and nothing may be read from it")
        return 1
    except P35Error as exc:
        os.close(saved_stdout_fd)
        log(f"FAIL: {exc}")
        log(
            "no CSV header and no CSV row are emitted, including rows already completed; "
            "this run is not a completed comparison"
        )
        return 1
    except BaseException:
        os.close(saved_stdout_fd)
        raise

    # Only a fully completed five-shape, four-candidate sweep, with every
    # correctness check already passed, reaches this line, and this is the only
    # place a CSV is ever written.
    _emit_on_saved_stdout(saved_stdout_fd, csv_text)
    log(
        f"emitted {EXPECTED_ROW_COUNT} non-publishable P3.5 rows (functional comparison "
        "evidence, not an experimental result and not a final conclusion)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
