#!/usr/bin/env python3
"""The experimental context that every run records in its metadata.json."""

import csv
import datetime as dt
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_FIELDS = ("uuid", "name", "driver_version", "pci.bus_id", "vbios_version",
              "clocks.max.sm", "clocks.max.memory", "power.limit", "ecc.mode.current")
PACKAGES = ("torch", "nvidia-cutlass-dsl", "cuda-python", "cuda-bindings", "nvidia-cublas")


def output(*command):
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                          check=True).stdout.strip()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def line_containing(text, word):
    return next(line.strip() for line in text.splitlines() if word in line)


def source_commit():
    """Require committed sources before acquisition and again before marking it complete."""
    if output("git", "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit the modified tracked files before acquiring measurements")
    return output("git", "rev-parse", "HEAD")


def complete(record):
    if source_commit() != record["git_commit"]:
        raise ValueError("the source commit changed during acquisition")
    record["completed_utc"] = utc_now()


def environment():
    """Source commit, GPU, CUDA, Nsight Compute, CUTLASS and Python package versions."""
    commit = source_commit()
    # run_gpu.sh exposes one physical GPU to the container and names it here.
    gpu = os.environ.get("BLACKWELL_GPU_UUID", "0")
    values = next(csv.reader([output("nvidia-smi", "-i", gpu,
                                     f"--query-gpu={','.join(GPU_FIELDS)}",
                                     "--format=csv,noheader,nounits")]))
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "git_commit": commit, "git_dirty": False,
        "gpu": {field.replace(".", "_"): value.strip() for field, value in zip(GPU_FIELDS, values)},
        "cuda_toolkit": line_containing(output("nvcc", "--version"), "release"),
        "nsight_compute": line_containing(output("ncu", "--version"), "Version"),
        "cutlass_commit": output("git", "-c", "safe.directory=/opt/cutlass", "-C", "/opt/cutlass",
                                 "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "packages": packages,
    }
