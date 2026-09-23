include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
CAMPAIGN_KIND ?= final
CAMPAIGN_ID ?=
CAMPAIGN_ROOT ?= runs
CAMPAIGN_NCU ?= $(if $(filter pilot,$(CAMPAIGN_KIND)),0,1)
CAMPAIGN_EXPERIMENTS ?=
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?= results
ANALYSIS_ONLY ?=
PROFILE_ID ?= gemm-profile-final
PROFILE_CACHE ?= cold
PRECISION_ID ?= precision-extended-final
ARCH ?= $(CUDA_ARCH)
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
BINARIES := $(addprefix build/memory_paths/,ldgsts tma) \
            $(addprefix build/umma_throughput/,umma_1sm umma_2sm umma_device_scaling)
BRIDGE := build/gemm_comparison/libcublaslt_bridge.so
PRECISION_BRIDGE := build/precision_comparison/libcublaslt_precision_bridge.so
export IMAGE_TAG

.DEFAULT_GOAL := help
.PHONY: help image build compile sass clean smoke campaign analyze precision \
	gemm-profile precision-extended test check-diagnostics

help:
	@echo "make image              Build the pinned CUDA/CuTe DSL image"
	@echo "make build              Compile the five benchmarks and both cuBLASLt bridges"
	@echo "make smoke              Run a short pilot of all four experiments"
	@echo "make campaign           Run one pilot or final campaign"
	@echo "make analyze            Summarize three final campaigns"
	@echo "make precision          Compare matched BF16, FP8 and NVFP4 GEMMs"
	@echo "make gemm-profile       Profile one P2 and one cuBLASLt launch per shape"
	@echo "make precision-extended Add cuBLASLt to the precision comparison"
	@echo "make test               Run the focused checks inside the pinned image"
	@echo "make check-diagnostics  Check both diagnostic output directories"
	@echo "make sass               Generate optional SASS disassemblies"

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
		$(if $(strip $(CAMPAIGN_EXPERIMENTS)),--experiments "$(CAMPAIGN_EXPERIMENTS)") \
		$(if $(filter 1 yes true,$(CAMPAIGN_NCU)),--with-ncu)

analyze:
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(CAMPAIGN_ROOT)/$(id)") \
		$(if $(strip $(ANALYSIS_ONLY)),--only "$(ANALYSIS_ONLY)") \
		--output "$(ANALYSIS_OUT)"

precision:
	scripts/run_gpu.sh python3 precision_comparison/precision_comparison.py \
		--output "$(ANALYSIS_OUT)"

gemm-profile: build
	scripts/run_gpu.sh python3 scripts/profile_gemm.py --cache-state "$(PROFILE_CACHE)" \
		--output "$(CAMPAIGN_ROOT)/$(PROFILE_ID)"

precision-extended: build
	scripts/run_gpu.sh python3 precision_comparison/precision_comparison.py --with-cublaslt \
		--output "$(CAMPAIGN_ROOT)/$(PRECISION_ID)"

test:
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		python3 -m unittest discover -s tests -v

check-diagnostics:
	python3 scripts/check_diagnostics.py --gemm-profile "$(CAMPAIGN_ROOT)/$(PROFILE_ID)" \
		--precision "$(CAMPAIGN_ROOT)/$(PRECISION_ID)"

clean:
	rm -rf build
