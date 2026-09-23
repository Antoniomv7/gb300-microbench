include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
CAMPAIGN_KIND ?= final
CAMPAIGN_ID ?=
CAMPAIGN_ROOT ?= runs
CAMPAIGN_NCU ?= $(if $(filter pilot,$(CAMPAIGN_KIND)),0,1)
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?=
PROFILE_ID ?=
PROFILE_CACHE ?=
GEMM_SUMMARY ?=
PRECISION_ID ?=
ARCH ?= $(CUDA_ARCH)
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
BINARIES := $(addprefix build/memory_paths/,ldgsts tma) \
            $(addprefix build/umma_throughput/,umma_1sm umma_2sm umma_device_scaling)
BRIDGE := build/gemm_comparison/libcublaslt_bridge.so
PRECISION_BRIDGE := build/precision_comparison/libcublaslt_precision_bridge.so
export IMAGE_TAG

# Fail before any GPU work when a required variable is empty.
require = $(if $(strip $($(1))),,$(error $(1) is required))

.DEFAULT_GOAL := help
.PHONY: help image build compile clean smoke campaign analyze precision-extended gemm-profile \
	check-diagnostics

help:
	@echo "make image              Build the pinned CUDA/CuTe DSL image"
	@echo "make build              Compile the five benchmarks and both cuBLASLt bridges"
	@echo "make smoke              Run a short pilot of the four campaign experiments"
	@echo "make campaign           Run one pilot or final campaign (CAMPAIGN_ID, CAMPAIGN_NCU)"
	@echo "make analyze            Summarize three final campaigns (FINAL_CAMPAIGNS, ANALYSIS_OUT)"
	@echo "make precision-extended Run Experiment V with its cuBLASLt baseline (PRECISION_ID)"
	@echo "make gemm-profile       Profile P2 and cuBLASLt with Nsight Compute"
	@echo "                        (PROFILE_CACHE=hot|cold, PROFILE_ID, GEMM_SUMMARY)"
	@echo "make check-diagnostics  Check a GEMM profile and a precision run (PROFILE_ID, PRECISION_ID)"

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
		make compile ARCH="$(CUDA_ARCH)"

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

smoke: build
	scripts/run_gpu.sh python3 scripts/run_campaign.py --kind pilot \
		--campaign-id "smoke-$$(date -u +%Y%m%dT%H%M%SZ)" --output-root "$(CAMPAIGN_ROOT)"

campaign: build
	scripts/run_gpu.sh python3 scripts/run_campaign.py --kind "$(CAMPAIGN_KIND)" \
		--output-root "$(CAMPAIGN_ROOT)" \
		$(if $(strip $(CAMPAIGN_ID)),--campaign-id "$(CAMPAIGN_ID)") \
		$(if $(filter 1 yes true,$(CAMPAIGN_NCU)),--with-ncu)

analyze:
	$(call require,ANALYSIS_OUT)
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(CAMPAIGN_ROOT)/$(id)") \
		--output "$(ANALYSIS_OUT)"

precision-extended: build
	$(call require,PRECISION_ID)
	scripts/run_gpu.sh python3 precision_comparison/precision_comparison.py \
		--output "$(CAMPAIGN_ROOT)/$(PRECISION_ID)"

gemm-profile: build
	$(call require,PROFILE_ID)
	$(call require,PROFILE_CACHE)
	$(call require,GEMM_SUMMARY)
	scripts/run_gpu.sh python3 scripts/profile_gemm.py --cache-state "$(PROFILE_CACHE)" \
		--gemm-summary "$(GEMM_SUMMARY)" --output "$(CAMPAIGN_ROOT)/$(PROFILE_ID)"

check-diagnostics:
	$(call require,PROFILE_ID)
	$(call require,PRECISION_ID)
	python3 scripts/check_diagnostics.py --gemm-profile "$(CAMPAIGN_ROOT)/$(PROFILE_ID)" \
		--precision "$(CAMPAIGN_ROOT)/$(PRECISION_ID)"

clean:
	rm -rf build
