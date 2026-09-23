include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
RUNS ?= runs
# Optional IDs of new directories under RUNS; by default a UTC timestamp makes them unique.
CAMPAIGN_ID ?=
PRECISION_ID ?=
PROFILE_ID ?=
# Inputs of the underlying analyze and gemm-profile commands, which final-study sets itself.
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?=
PROFILE_CACHE ?=
GEMM_SUMMARY ?=
ARCH ?= $(CUDA_ARCH)
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
BINARIES := $(addprefix build/memory_paths/,ldgsts tma) \
            $(addprefix build/umma_throughput/,umma_1sm umma_2sm umma_device_scaling)
BRIDGE := build/gemm_comparison/libcublaslt_bridge.so
PRECISION_BRIDGE := build/precision_comparison/libcublaslt_precision_bridge.so
STAMP = $$(date -u +%Y%m%dT%H%M%SZ)
CHECK = python3 scripts/check_diagnostics.py
export IMAGE_TAG

# Fail before any GPU work when a required variable is empty.
require = $(if $(strip $($(1))),,$(error $(1) is required))
# $(1): new run ID; $(2): experiments, or empty for all four. Each run is checked independently.
run_campaign = id="$(1)"; set -e; \
	scripts/run_gpu.sh python3 scripts/run_campaign.py $(if $(2),--experiments $(2)) \
		--output-root "$(RUNS)" --campaign-id "$$id"; \
	$(CHECK) --campaign "$(RUNS)/$$id"

.DEFAULT_GOAL := help
.PHONY: help image build compile clean exp1-memory exp2-umma exp3-scaling exp4-gemm precision \
	final-study campaign analyze gemm-profile

help:
	@echo "Set BLACKWELL_GPU_INDEX, then (final parameters, one new directory under $(RUNS)/ each):"
	@echo "  make exp1-memory   I    LDGSTS versus TMA, with its 6 NCU DRAM captures"
	@echo "  make exp2-umma     II   isolated 1-SM versus 2-SM UMMA, with its 2 NCU SM-clock captures"
	@echo "  make exp3-scaling  III  whole-device UMMA scaling, with nvidia-smi clock telemetry"
	@echo "  make exp4-gemm     IV   BF16 CuTe DSL variants versus cuBLASLt"
	@echo "  make precision     V    BF16, FP8 and NVFP4 with CuTe DSL and cuBLASLt"
	@echo "  make final-study        I-IV three times, V once, hot-cache GEMM profile, archive"
	@echo "Setup: make image; make build. Underlying commands for diagnosis:"
	@echo "  make campaign [CAMPAIGN_ID=id]"
	@echo "  make analyze FINAL_CAMPAIGNS=\"id1 id2 id3\" ANALYSIS_OUT=directory"
	@echo "  make gemm-profile PROFILE_CACHE=hot|cold GEMM_SUMMARY=analysis/gemm_comparison.csv"
	@echo "  python3 scripts/check_diagnostics.py --help"

image:
	docker build --platform "$(CUDA_IMAGE_PLATFORM)" \
		--build-arg BASE_IMAGE="$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" \
		--build-arg CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		--build-arg PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		--build-arg PYTORCH_INDEX_URL="$(PYTORCH_INDEX_URL)" \
		--build-arg CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		--build-arg MAX_BUILD_JOBS="$(MAX_BUILD_JOBS)" -t "$(IMAGE_TAG)" .

build:
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		make compile ARCH="$(ARCH)"

compile: $(BINARIES) $(BRIDGE) $(PRECISION_BRIDGE)

build/memory_paths/%: memory_paths/%.cu memory_paths/memory_common.cuh benchmark_common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $<

build/umma_throughput/%: umma_throughput/%.cu umma_throughput/umma_common.cuh benchmark_common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -arch=$(VIRTUAL_ARCH) -code=$(ARCH) -o $@ $<

$(BRIDGE): gemm_comparison/cublaslt_bridge.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -arch=$(VIRTUAL_ARCH) \
		-code=$(ARCH) -o $@ $< -lcublasLt -lcudart

$(PRECISION_BRIDGE): precision_comparison/cublaslt_precision_bridge.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -arch=$(VIRTUAL_ARCH) \
		-code=$(ARCH) -o $@ $< -lcublasLt -lcudart

exp1-memory: build
	$(call run_campaign,exp1-memory-$(STAMP),memory_paths)

exp2-umma: build
	$(call run_campaign,exp2-umma-$(STAMP),umma_throughput)

exp3-scaling: build
	$(call run_campaign,exp3-scaling-$(STAMP),umma_device_scaling)

exp4-gemm: build
	$(call run_campaign,exp4-gemm-$(STAMP),gemm_comparison)

precision: build
	id="$(or $(PRECISION_ID),precision-$(STAMP))"; set -e; \
	scripts/run_gpu.sh python3 precision_comparison/precision_comparison.py \
		--output "$(RUNS)/$$id"; \
	$(CHECK) --precision "$(RUNS)/$$id"

final-study:
	python3 scripts/final_study.py --runs "$(RUNS)"

campaign: build
	$(call run_campaign,$(or $(CAMPAIGN_ID),campaign-$(STAMP)),)

analyze:
	$(call require,ANALYSIS_OUT)
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(RUNS)/$(id)") \
		--output "$(ANALYSIS_OUT)"

gemm-profile: build
	$(call require,PROFILE_CACHE)
	$(call require,GEMM_SUMMARY)
	id="$(or $(PROFILE_ID),gemm-profile-$(PROFILE_CACHE)-$(STAMP))"; set -e; \
	scripts/run_gpu.sh python3 scripts/profile_gemm.py --cache-state "$(PROFILE_CACHE)" \
		--gemm-summary "$(GEMM_SUMMARY)" --output "$(RUNS)/$$id"; \
	$(CHECK) --gemm-profile "$(RUNS)/$$id"

clean:
	rm -rf build
