include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
ARCH ?= $(CUDA_ARCH)
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
BINARIES := $(addprefix build/memory_paths/,ldgsts tma) \
            $(addprefix build/umma_throughput/,umma_1sm umma_2sm umma_device_scaling)
BRIDGES := build/gemm_comparison/libcublaslt_bridge.so \
           build/precision_comparison/libcublaslt_precision_bridge.so
# Runs a command in the container on the idle GPU chosen by BLACKWELL_GPU_INDEX.
GPU := scripts/run_gpu.sh
# One UTC timestamp per make invocation names every new directory under runs/.
STAMP := $(shell date -u +%Y%m%dT%H%M%SZ)
# Inputs of the standalone analysis and GEMM-profile targets.
CAMPAIGNS ?=
PROFILE_CACHE ?= hot
GEMM_SUMMARY ?=
STUDY ?=
export IMAGE_TAG

.DEFAULT_GOAL := help
.PHONY: help image build compile clean exp1-memory exp2-umma exp3-scaling exp4-gemm precision \
	final-study campaign analyze gemm-profile regenerate

help:
	@echo "Setup: make image, then export BLACKWELL_GPU_INDEX=<index of an idle B300>."
	@echo "Each target builds as needed and writes one new directory under runs/:"
	@echo "  make exp1-memory   I    LDGSTS versus TMA, with its 6 NCU DRAM captures"
	@echo "  make exp2-umma     II   isolated 1-SM versus 2-SM UMMA, with its 2 NCU SM-clock captures"
	@echo "  make exp3-scaling  III  whole-device UMMA scaling, with nvidia-smi clock telemetry"
	@echo "  make exp4-gemm     IV   BF16 CuTe DSL variants versus cuBLASLt"
	@echo "  make precision     V    BF16, FP8 and NVFP4 with CuTe DSL and cuBLASLt"
	@echo "  make final-study        I-IV three times, V, hot-cache GEMM profile, all summaries"
	@echo "Underlying steps:"
	@echo "  make campaign                                   I-IV once"
	@echo "  make analyze CAMPAIGNS=\"dir1 dir2 dir3\"          I-IV summaries (CPU only)"
	@echo "  make gemm-profile PROFILE_CACHE=hot|cold GEMM_SUMMARY=.../gemm_comparison.csv"
	@echo "  make regenerate STUDY=study-dir                 results/ from a study (CPU only)"

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

compile: $(BINARIES) $(BRIDGES)

build/memory_paths/%: memory_paths/%.cu memory_paths/memory_common.cuh benchmark_common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $<

build/umma_throughput/%: umma_throughput/%.cu umma_throughput/umma_common.cuh benchmark_common.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -arch=$(VIRTUAL_ARCH) -code=$(ARCH) -o $@ $<

build/gemm_comparison/libcublaslt_bridge.so: gemm_comparison/cublaslt_bridge.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -arch=$(VIRTUAL_ARCH) \
		-code=$(ARCH) -o $@ $< -lcublasLt -lcudart

build/precision_comparison/libcublaslt_precision_bridge.so: \
		precision_comparison/cublaslt_precision_bridge.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared -arch=$(VIRTUAL_ARCH) \
		-code=$(ARCH) -o $@ $< -lcublasLt -lcudart

exp1-memory: build
	$(GPU) python3 scripts/run_campaign.py --experiments memory_paths \
		--output runs/exp1-memory-$(STAMP)

exp2-umma: build
	$(GPU) python3 scripts/run_campaign.py --experiments umma_throughput \
		--output runs/exp2-umma-$(STAMP)

exp3-scaling: build
	$(GPU) python3 scripts/run_campaign.py --experiments umma_device_scaling \
		--output runs/exp3-scaling-$(STAMP)

exp4-gemm: build
	$(GPU) python3 scripts/run_campaign.py --experiments gemm_comparison \
		--output runs/exp4-gemm-$(STAMP)

precision: build
	$(GPU) python3 precision_comparison/precision_comparison.py --output runs/precision-$(STAMP)
	python3 analysis/analyze.py --precision runs/precision-$(STAMP) \
		--output runs/precision-$(STAMP)/analysis

campaign: build
	$(GPU) python3 scripts/run_campaign.py --output runs/campaign-$(STAMP)

analyze:
	python3 analysis/analyze.py --campaigns $(CAMPAIGNS) --output runs/analysis-$(STAMP)

gemm-profile: build
	@test -n "$(GEMM_SUMMARY)" || { echo "set GEMM_SUMMARY to a gemm_comparison.csv" >&2; exit 2; }
	$(GPU) python3 scripts/profile_gemm.py --cache-state $(PROFILE_CACHE) \
		--output runs/gemm-profile-$(PROFILE_CACHE)-$(STAMP)
	python3 analysis/analyze.py --gemm-profile runs/gemm-profile-$(PROFILE_CACHE)-$(STAMP) \
		--cache-state $(PROFILE_CACHE) --gemm-summary $(GEMM_SUMMARY) \
		--output runs/gemm-profile-$(PROFILE_CACHE)-$(STAMP)/analysis

# The binaries are built once; every step runs in the container except the CPU-only analysis.
final-study: FINAL := runs/study-$(STAMP)
final-study: build
	$(GPU) python3 scripts/run_campaign.py --output $(FINAL)/campaign-1
	$(GPU) python3 scripts/run_campaign.py --output $(FINAL)/campaign-2
	$(GPU) python3 scripts/run_campaign.py --output $(FINAL)/campaign-3
	$(GPU) python3 precision_comparison/precision_comparison.py --output $(FINAL)/precision
	$(GPU) python3 scripts/profile_gemm.py --cache-state hot --output $(FINAL)/gemm-profile-hot
	python3 analysis/analyze.py --study $(FINAL) --output $(FINAL)/analysis

regenerate:
	python3 analysis/analyze.py --study $(STUDY) --output runs/regenerated-$(STAMP)

clean:
	rm -rf build
