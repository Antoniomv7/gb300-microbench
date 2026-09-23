#!/usr/bin/env python3
"""Record the GPU, software and repository state behind a diagnostic run."""

import csv
import ctypes
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_FIELDS = ("uuid", "name", "driver_version", "pci.bus_id", "vbios_version",
              "clocks.max.sm", "clocks.max.memory", "power.limit", "ecc.mode.current")
PACKAGES = ("torch", "nvidia-cutlass-dsl", "cuda-python", "cuda-bindings", "nvidia-cublas")


def command_output(command, cwd=ROOT):
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def gpu_identity():
    # run_gpu.sh exposes one physical GPU to the container and names it here.
    gpu = os.environ.get("BLACKWELL_GPU_UUID", "0")
    output = command_output(["nvidia-smi", "-i", gpu, f"--query-gpu={','.join(GPU_FIELDS)}",
                             "--format=csv,noheader,nounits"])
    values = next(csv.reader(output.splitlines()))
    return {field.replace(".", "_"): value.strip() for field, value in zip(GPU_FIELDS, values)}


def repository_state(sources=()):
    """Identify the commit, uncommitted changes and the exact source files behind a run."""
    return {"commit": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--porcelain"]).splitlines(),
            "source_sha256": {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
                              for source in sources}}


def tracked_changes(repository):
    """Return the status lines of modified tracked files; untracked files do not change sources."""
    return [line for line in repository.get("status", []) if not line.startswith("??")]


def ncu_version():
    output = command_output([os.environ.get("NCU_BINARY", "ncu"), "--version"])
    return next((line.strip() for line in output.splitlines() if "Version" in line), output.strip())


def pinned_versions():
    pairs = (line.split("=", 1) for line in (ROOT / "VERSIONS.env").read_text().splitlines()
             if "=" in line and not line.startswith("#"))
    return {key.strip(): value.strip() for key, value in pairs}


def loaded_library(fragment):
    """Return the shared objects mapped into this process whose path contains fragment."""
    with open("/proc/self/maps", encoding="utf-8") as maps:
        return sorted({line.split()[-1] for line in maps
                       if fragment in line and line.split()[-1].startswith("/")})


def cuda_driver_version():
    try:
        version = ctypes.c_int()
        status = ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version))
    except OSError:
        return None
    return None if status else version.value


def software_versions():
    versions = {"python": platform.python_version(), "executable": sys.executable,
                "cuda_driver_api": cuda_driver_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        versions["cutlass_checkout"] = command_output(
            ["git", "-c", "safe.directory=/opt/cutlass", "-C", "/opt/cutlass", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        versions["cutlass_checkout"] = None
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
    except ImportError:
        pass
    return versions
