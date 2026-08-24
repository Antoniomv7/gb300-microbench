include VERSIONS.env
include PHASE3_VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
MEMORY_OUT ?= runs/memory_paths.csv
UMMA_OUT ?= runs/umma_throughput.csv
GEMM_OUT ?= runs/gemm_comparison.csv
SMOKE_ID := smoke-$(shell date -u +%Y%m%dT%H%M%SZ)
SMOKE_DIR ?= runs/$(SMOKE_ID)
CAMPAIGN_KIND ?= final
CAMPAIGN_ID ?=
CAMPAIGN_ROOT ?= runs
CAMPAIGN_NCU ?=
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?= results/new
NCU_GATE_ID := ncu-gate-$(shell date -u +%Y%m%dT%H%M%SZ)
NCU_GATE_DIR ?= runs/$(NCU_GATE_ID)

# Supplementary UMMA launch-scale experiment. It has its own campaign root,
# its own raw CSV schema and its own results location, so it can never be
# mixed with the closed three-experiment population under runs/<UTC-ID>/ and
# results/new.
UMMA_SCALING_ROOT ?= runs/umma_device_scaling
UMMA_SCALING_KIND ?= final
UMMA_SCALING_ID ?=
UMMA_SCALING_FINALS ?=
UMMA_SCALING_ANALYSIS_OUT ?= results/umma_device_scaling/latest
UMMA_SCALING_SMOKE_ID := smoke-$(shell date -u +%Y%m%dT%H%M%SZ)
UMMA_SCALING_SMOKE_DIR ?= $(UMMA_SCALING_ROOT)/$(UMMA_SCALING_SMOKE_ID)
export IMAGE_TAG

.DEFAULT_GOAL := help
.PHONY: help image build sass smoke smoke-verify ncu-check memory-run umma-run gemm-run campaign analyze clean \
        umma-scaling-build umma-scaling-sass umma-scaling-check umma-scaling-smoke \
        umma-scaling-campaign umma-scaling-analyze

help:
	@echo "make image       Build the pinned CUDA/CuTe DSL image"
	@echo "make build       Compile the four CUDA microbenchmarks"
	@echo "make sass        Disassemble locally rebuilt binaries"
	@echo "make clean       Remove build/ so the next build cannot reuse a stale binary"
	@echo "make memory-run  Run the 18-case memory sweep (requires BLACKWELL_GPU_INDEX)"
	@echo "make umma-run    Run the 24-case UMMA sweep (requires BLACKWELL_GPU_INDEX)"
	@echo "make gemm-run    Run five shapes x four GEMM candidates (requires BLACKWELL_GPU_INDEX)"
	@echo "make smoke       Run short GPU validation for all three experiments"
	@echo "make smoke-verify Re-check SMOKE_DIR against the raw CSV contract, without a GPU"
	@echo "make ncu-check   Require one real, parsed Nsight Compute capture"
	@echo "make campaign    Freeze one pilot/final campaign under runs/<UTC-ID>/"
	@echo "make analyze     Aggregate exactly three explicit final campaigns"
	@echo ""
	@echo "Supplementary UMMA launch-scale experiment (isolated vs device scale):"
	@echo "make umma-scaling-build     Compile only the device-scaling binary"
	@echo "make umma-scaling-sass      Disassemble the device-scaling binary"
	@echo "make umma-scaling-check     GPU-free syntax, CLI and analyzer tests"
	@echo "make umma-scaling-smoke     Self-test plus a short four-configuration GPU run"
	@echo "make umma-scaling-campaign  Freeze one pilot/final supplementary campaign"
	@echo "make umma-scaling-analyze   Aggregate exactly three final supplementary campaigns"
	@echo ""
	@echo "MEMORY_OUT, UMMA_OUT, GEMM_OUT, SMOKE_DIR and ANALYSIS_OUT are all relative"
	@echo "to this repository root. GEMM_OUT included: the '../' that gemm_comparison/"
	@echo "Makefile expects is added here, at the point of use, not in the value."

image:
	docker build --platform "$(CUDA_IMAGE_PLATFORM)" \
		--build-arg BASE_IMAGE="$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" \
		--build-arg CUTLASS_VERSION="$(CUTLASS_VERSION)" \
		--build-arg CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		--build-arg PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		--build-arg PYTORCH_INDEX_URL="$(PYTORCH_INDEX_URL)" \
		--build-arg PYTORCH_CUDA_VERSION="$(PYTORCH_CUDA_VERSION)" \
		--build-arg CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		--build-arg CUDA_BINDINGS_VERSION="$(CUDA_BINDINGS_VERSION)" \
		--build-arg MAX_BUILD_JOBS="$(MAX_BUILD_JOBS)" \
		-t "$(IMAGE_TAG)" .

build:
	@mkdir -p build/memory_paths build/umma_throughput build/sass
	docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
		--user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C memory_paths ARCH="$(CUDA_ARCH)" && make -C umma_throughput ARCH="$(CUDA_ARCH)"'

sass: build
	docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
		--user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C memory_paths ARCH="$(CUDA_ARCH)" sass && make -C umma_throughput ARCH="$(CUDA_ARCH)" sass'

clean:
	rm -rf build

memory-run: build
	@test ! -e "$(MEMORY_OUT)" || { echo "memory-run output already exists: $(MEMORY_OUT)" >&2; exit 2; }
	python3 memory_paths/benchmark.py --run-kind benchmark --output "$(MEMORY_OUT)"

umma-run: build
	@test ! -e "$(UMMA_OUT)" || { echo "umma-run output already exists: $(UMMA_OUT)" >&2; exit 2; }
	python3 umma_throughput/benchmark.py --run-kind benchmark --output "$(UMMA_OUT)"

gemm-run:
	@test ! -e "$(GEMM_OUT)" || { echo "gemm-run output already exists: $(GEMM_OUT)" >&2; exit 2; }
	@mkdir -p "$(dir $(GEMM_OUT))"
	scripts/run_gpu.sh make -C gemm_comparison run ARCH="$(CUDA_ARCH)" OUTPUT="../$(GEMM_OUT)" WARMUP=2 ITERATIONS=10

smoke: build
	@test ! -e "$(SMOKE_DIR)" || { echo "smoke output already exists: $(SMOKE_DIR)" >&2; exit 2; }
	@mkdir -p "$(SMOKE_DIR)"
	python3 memory_paths/benchmark.py --run-kind smoke --output "$(SMOKE_DIR)/memory_paths.csv"
	python3 umma_throughput/benchmark.py --run-kind smoke --output "$(SMOKE_DIR)/umma_throughput.csv"
	scripts/run_gpu.sh make -C gemm_comparison run ARCH="$(CUDA_ARCH)" \
		OUTPUT="../$(SMOKE_DIR)/gemm_comparison.csv" WARMUP=1 ITERATIONS=1
	python3 -c "$$SMOKE_VERIFY_PY" "$(SMOKE_DIR)"

smoke-verify:
	@test -d "$(SMOKE_DIR)" || { echo "smoke-verify: pass SMOKE_DIR=runs/smoke-<id> (the directory make smoke created)" >&2; exit 2; }
	python3 -c "$$SMOKE_VERIFY_PY" "$(SMOKE_DIR)"

ncu-check: build
	@test ! -e "$(NCU_GATE_DIR)" || { echo "ncu-check output already exists: $(NCU_GATE_DIR)" >&2; exit 2; }
	@mkdir -p "$(NCU_GATE_DIR)"
	python3 scripts/ncu_capture.py --campaign-dir "$(NCU_GATE_DIR)" \
		--case 00_ldgsts_s2_bif16 --require-complete

campaign: build
	python3 scripts/run_campaign.py --kind "$(CAMPAIGN_KIND)" \
		--output-root "$(CAMPAIGN_ROOT)" $(if $(strip $(CAMPAIGN_ID)),--campaign-id "$(CAMPAIGN_ID)") \
		$(if $(strip $(CAMPAIGN_NCU)),--with-ncu)

analyze:
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three campaign IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(CAMPAIGN_ROOT)/$(id)") \
		--output "$(ANALYSIS_OUT)"

# --- Supplementary UMMA launch-scale experiment -----------------------------
# `build` and `sass` above stay exactly as they were: the device-scaling binary
# is a separate goal of umma_throughput/Makefile, so the frozen campaign
# contract's build behaviour is unchanged.

umma-scaling-build:
	@mkdir -p build/umma_throughput build/sass
	docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
		--user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C umma_throughput ARCH="$(CUDA_ARCH)" scaling'

umma-scaling-sass: umma-scaling-build
	docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
		--user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C umma_throughput ARCH="$(CUDA_ARCH)" scaling-sass'

umma-scaling-check:
	python3 -m py_compile umma_throughput/device_scaling.py \
		scripts/run_umma_scaling_campaign.py analysis/analyze_umma_device_scaling.py \
		tests/test_umma_device_scaling.py
	python3 analysis/analyze_umma_device_scaling.py --help >/dev/null
	python3 tests/test_umma_device_scaling.py

umma-scaling-smoke: umma-scaling-build
	@test ! -e "$(UMMA_SCALING_SMOKE_DIR)" || { \
		echo "smoke output already exists: $(UMMA_SCALING_SMOKE_DIR)" >&2; exit 2; }
	@mkdir -p "$(UMMA_SCALING_SMOKE_DIR)"
	scripts/run_gpu.sh build/umma_throughput/umma_device_scaling --self-test
	python3 umma_throughput/device_scaling.py --run-kind smoke \
		--output "$(UMMA_SCALING_SMOKE_DIR)/umma_device_scaling.csv"

umma-scaling-campaign: umma-scaling-build
	python3 scripts/run_umma_scaling_campaign.py --kind "$(UMMA_SCALING_KIND)" \
		--output-root "$(UMMA_SCALING_ROOT)" \
		$(if $(strip $(UMMA_SCALING_ID)),--campaign-id "$(UMMA_SCALING_ID)")

umma-scaling-analyze:
	@test "$(words $(UMMA_SCALING_FINALS))" -eq 3 || { \
		echo "UMMA_SCALING_FINALS must contain exactly three campaign IDs" >&2; exit 2; }
	python3 analysis/analyze_umma_device_scaling.py \
		$(foreach id,$(UMMA_SCALING_FINALS),--campaign "$(UMMA_SCALING_ROOT)/$(id)") \
		--output "$(UMMA_SCALING_ANALYSIS_OUT)"

define SMOKE_VERIFY_PY
import csv, sys
from pathlib import Path

COMMON = {"run_kind", "correctness", "gpu_uuid", "git_commit", "git_dirty"}
SPECS = {
    "memory_paths.csv": (
        COMMON | {"method", "sample_index", "stages", "bytes_in_flight_per_sm",
                  "kernel_time_ms", "effective_gbps", "mismatches"},
        ("method", "stages", "bytes_in_flight_per_sm"), 18, "OK"),
    "umma_throughput.csv": (
        COMMON | {"publishable", "method", "sample_index", "cta_group", "n", "depth",
                  "elapsed_cycles", "flops_per_cycle", "mismatches"},
        ("method", "n", "depth"), 24, "OK"),
    "gemm_comparison.csv": (
        COMMON | {"schema_version", "publishable", "shape_index", "shape_id",
                  "candidate_index", "method", "variant", "kernel_time_ms", "tflops",
                  "throughput_ratio_vs_cublaslt", "gap_to_cublaslt_pct",
                  "best_cutedsl_variant"},
        ("shape_index", "candidate_index"), 20, "PASS"),
}

root = Path(sys.argv[1])
errors = []
identities = set()
for name, (required, keys, configurations, passed) in SPECS.items():
    path = root / name
    if not path.is_file():
        errors.append(name + ": missing")
        continue
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = sorted(required - set(fields))
    if missing:
        errors.append("{}: missing columns {}".format(name, missing))
        continue
    if not rows:
        errors.append(name + ": no data rows")
        continue
    groups = {}
    for number, row in enumerate(rows, 2):
        where = "{}:{}".format(name, number)
        if row.get("run_kind") != "smoke":
            errors.append(where + ": run_kind is not smoke")
        if row.get("correctness") != passed:
            errors.append(where + ": correctness is not " + passed)
        if row.get("git_dirty") != "false":
            errors.append(where + ": git_dirty is not false")
        if not (row.get("gpu_uuid") or "").startswith("GPU-"):
            errors.append(where + ": malformed or empty gpu_uuid")
        identities.add((row.get("gpu_uuid"), row.get("git_commit")))
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row.get("sample_index"))
    if len(groups) != configurations:
        errors.append("{}: {} configurations, expected {}".format(name, len(groups), configurations))
    for key, samples in groups.items():
        if samples[0] is None:
            if len(samples) != 1:
                errors.append("{}: {} duplicate rows for {}".format(name, len(samples), key))
            continue
        try:
            indexes = sorted(int(value) for value in samples)
        except (TypeError, ValueError):
            errors.append("{}: non-integer sample_index for {}".format(name, key))
            continue
        if indexes != list(range(len(indexes))):
            errors.append("{}: {} sample_index is not exactly 0..{}".format(
                name, key, len(indexes) - 1))
if len(identities) > 1:
    errors.append("mixed gpu_uuid/git_commit across the three CSVs: {}".format(
        sorted(map(str, identities))))
for message in errors[:20]:
    print("smoke-verify: " + message, file=sys.stderr)
if errors:
    print("smoke-verify: FAILED, {} problem(s) in {}".format(len(errors), root), file=sys.stderr)
    raise SystemExit(2)
print("smoke-verify: OK {}".format(root), file=sys.stderr)
endef
export SMOKE_VERIFY_PY
