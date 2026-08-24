include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
CAMPAIGN_KIND ?= final
CAMPAIGN_ID ?=
CAMPAIGN_ROOT ?= runs
CAMPAIGN_NCU ?= $(if $(filter pilot,$(CAMPAIGN_KIND)),0,1)
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?= results/generated
ARCH ?= $(CUDA_ARCH)
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
BINARIES := $(addprefix build/memory_paths/,ldgsts tma) \
            $(addprefix build/umma_throughput/,umma_1sm umma_2sm umma_device_scaling)
BRIDGE := build/gemm_comparison/libcublaslt_bridge.so
export IMAGE_TAG

.DEFAULT_GOAL := help
.PHONY: help image build compile sass clean smoke campaign analyze

help:
	@echo "make image      Build the pinned CUDA/CuTe DSL image"
	@echo "make build      Compile the five benchmarks and cuBLASLt bridge"
	@echo "make smoke      Run a short pilot of all four experiments"
	@echo "make campaign   Run one pilot or final campaign"
	@echo "make analyze    Summarize three final campaigns"
	@echo "make sass       Generate optional SASS disassemblies"

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

compile: $(BINARIES) $(BRIDGE)

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

sass: build
	@mkdir -p build/sass
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'for binary in build/memory_paths/* build/umma_throughput/*; do \
			cuobjdump --dump-sass "$$binary" > "build/sass/$$(basename "$$binary").sass"; done'

smoke: build
	scripts/run_gpu.sh python3 scripts/run_campaign.py --kind pilot \
		--campaign-id "smoke-$$(date -u +%Y%m%dT%H%M%SZ)" --output-root "$(CAMPAIGN_ROOT)"

campaign: build
	scripts/run_gpu.sh python3 scripts/run_campaign.py --kind "$(CAMPAIGN_KIND)" \
		--output-root "$(CAMPAIGN_ROOT)" \
		$(if $(strip $(CAMPAIGN_ID)),--campaign-id "$(CAMPAIGN_ID)") \
		$(if $(filter 1 yes true,$(CAMPAIGN_NCU)),--with-ncu)

analyze:
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(CAMPAIGN_ROOT)/$(id)") \
		--output "$(ANALYSIS_OUT)"

clean:
	rm -rf build
