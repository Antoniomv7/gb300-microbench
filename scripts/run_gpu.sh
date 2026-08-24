#!/usr/bin/env bash
set -euo pipefail

fail() { echo "run_gpu: $*" >&2; exit 2; }

[ "$#" -gt 0 ] || fail "usage: BLACKWELL_GPU_INDEX=<index> $0 <command>"
[[ "${BLACKWELL_GPU_INDEX:-}" =~ ^[0-9]+$ ]] || fail "set BLACKWELL_GPU_INDEX"

gpu_uuid="$(nvidia-smi -i "${BLACKWELL_GPU_INDEX}" \
    --query-gpu=uuid --format=csv,noheader)" || fail "cannot identify the selected GPU"
[[ "${gpu_uuid}" == GPU-* ]] || fail "invalid GPU UUID: ${gpu_uuid}"

processes="$(nvidia-smi -i "${gpu_uuid}" \
    --query-compute-apps=pid --format=csv,noheader)" || fail "cannot check GPU activity"
[ -z "${processes//[[:space:]]/}" ] || fail "GPU ${gpu_uuid} is already in use"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
echo "run_gpu: GPU ${BLACKWELL_GPU_INDEX} (${gpu_uuid})" >&2

exec docker run --rm --gpus "device=${gpu_uuid}" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e "BLACKWELL_GPU_UUID=${gpu_uuid}" \
    -v "${repo_root}:/workspace" -w /workspace \
    "${IMAGE_TAG:-gb300-microbench:latest}" "$@"
