#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
    echo "run_gpu: ERROR: $*" >&2
    exit 2
}

[ "$#" -gt 0 ] || fail "usage: BLACKWELL_GPU_INDEX=<index> $0 <command> [args...]"
[[ "${BLACKWELL_GPU_INDEX:-}" =~ ^[0-9]+$ ]] \
    || fail "BLACKWELL_GPU_INDEX must be set to one physical GPU index"

command -v docker >/dev/null 2>&1 || fail "docker is not installed"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is not installed"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${IMAGE_TAG:-gb300-microbench:latest}"

gpu_table="$(nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader)" \
    || fail "cannot query GPUs"
row="$(awk -F', *' -v idx="${BLACKWELL_GPU_INDEX}" \
    '$1 == idx {print; matches++} END {if (matches != 1) exit 1}' <<<"${gpu_table}")" \
    || fail "GPU index ${BLACKWELL_GPU_INDEX} does not resolve uniquely"

gpu_uuid="$(awk -F', *' '{print $2}' <<<"${row}")"
gpu_name="$(awk -F', *' '{print $3}' <<<"${row}")"
driver="$(awk -F', *' '{print $4}' <<<"${row}")"
[[ "${gpu_uuid}" =~ ^GPU-[0-9a-fA-F-]+$ ]] || fail "unexpected GPU UUID"

apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${gpu_uuid}")" \
    || fail "cannot verify whether ${gpu_uuid} is idle"
if grep -qiE 'insufficient|not supported|unknown|error|n/a' <<<"${apps}"; then
    fail "compute-process visibility is incomplete for ${gpu_uuid}; refusing to share the device"
fi
[ -z "${apps//[[:space:]]/}" ] || fail "GPU ${gpu_uuid} has active compute processes"

echo "run_gpu: index=${BLACKWELL_GPU_INDEX} uuid=${gpu_uuid} name='${gpu_name}' driver=${driver}" >&2

guard='
set -euo pipefail
mapfile -t visible < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ "${#visible[@]}" -eq 1 ] || { echo "expected exactly one visible GPU" >&2; exit 66; }
[ "${visible[0]}" = "${EXPECTED_GPU_UUID}" ] || { echo "GPU UUID mismatch" >&2; exit 66; }
exec "$@"
'

exec docker run --rm \
    --gpus "device=${gpu_uuid}" \
    --user "$(id -u):$(id -g)" \
    --network none \
    --security-opt no-new-privileges \
    --cap-drop ALL \
    --entrypoint /bin/bash \
    -e HOME=/tmp \
    -e EXPECTED_GPU_UUID="${gpu_uuid}" \
    -v "${repo_root}:/workspace" \
    -w /workspace \
    "${image_tag}" -c "${guard}" guard "$@"
