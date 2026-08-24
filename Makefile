include VERSIONS.env

IMAGE_TAG ?= gb300-microbench:latest
CAMPAIGN_KIND ?= final
CAMPAIGN_ID ?=
CAMPAIGN_ROOT ?= runs
CAMPAIGN_NCU ?= 1
FINAL_CAMPAIGNS ?=
ANALYSIS_OUT ?= results/generated
SMOKE_DIR ?= runs/smoke-$(shell date -u +%Y%m%dT%H%M%SZ)
export IMAGE_TAG

.DEFAULT_GOAL := help
.PHONY: help image build sass clean smoke memory-run umma-run umma-scaling-run \
        gemm-run campaign analyze profile ncu-check

help:
	@echo "make image             Build the reproducible CUDA/CuTe DSL image"
	@echo "make build             Compile all five CUDA microbenchmarks"
	@echo "make sass              Regenerate the five optional SASS disassemblies"
	@echo "make smoke             Run a short check of all four experiments"
	@echo "make campaign          Run one pilot or final four-experiment campaign"
	@echo "make analyze           Aggregate three final campaigns and plot all results"
	@echo "make profile           Profile an existing CAMPAIGN_DIR with Nsight Compute"
	@echo "make ncu-check         Check one real Nsight Compute capture"
	@echo "make clean             Remove generated binaries"

image:
	docker build --platform "$(CUDA_IMAGE_PLATFORM)" \
		--build-arg BASE_IMAGE="$(CUDA_IMAGE)@$(CUDA_IMAGE_DIGEST)" \
		--build-arg CUTLASS_COMMIT="$(CUTLASS_COMMIT)" \
		--build-arg PYTORCH_VERSION="$(PYTORCH_VERSION)" \
		--build-arg PYTORCH_INDEX_URL="$(PYTORCH_INDEX_URL)" \
		--build-arg CUDA_PYTHON_VERSION="$(CUDA_PYTHON_VERSION)" \
		--build-arg MAX_BUILD_JOBS="$(MAX_BUILD_JOBS)" \
		-t "$(IMAGE_TAG)" .

build:
	@mkdir -p build/memory_paths build/umma_throughput build/sass
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C memory_paths ARCH="$(CUDA_ARCH)" && make -C umma_throughput ARCH="$(CUDA_ARCH)"'

sass: build
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp \
		-v "$(CURDIR):/workspace" -w /workspace "$(IMAGE_TAG)" \
		bash -c 'make -C memory_paths ARCH="$(CUDA_ARCH)" sass && make -C umma_throughput ARCH="$(CUDA_ARCH)" sass'

clean:
	rm -rf build

memory-run: build
	python3 memory_paths/benchmark.py --run-kind benchmark --output runs/memory_paths.csv

umma-run: build
	python3 umma_throughput/benchmark.py --run-kind benchmark --output runs/umma_throughput.csv

umma-scaling-run: build
	python3 umma_throughput/benchmark.py --device-scaling --run-kind benchmark \
		--output runs/umma_device_scaling.csv

gemm-run:
	scripts/run_gpu.sh make -C gemm_comparison run ARCH="$(CUDA_ARCH)" \
		OUTPUT=../runs/gemm_comparison.csv WARMUP=2 ITERATIONS=10

smoke: build
	@mkdir -p "$(SMOKE_DIR)"
	python3 memory_paths/benchmark.py --run-kind smoke --output "$(SMOKE_DIR)/memory_paths.csv"
	python3 umma_throughput/benchmark.py --run-kind smoke --output "$(SMOKE_DIR)/umma_throughput.csv"
	python3 umma_throughput/benchmark.py --device-scaling --run-kind smoke \
		--output "$(SMOKE_DIR)/umma_device_scaling.csv"
	scripts/run_gpu.sh make -C gemm_comparison run ARCH="$(CUDA_ARCH)" \
		OUTPUT="../$(SMOKE_DIR)/gemm_comparison.csv" WARMUP=1 ITERATIONS=1

campaign: build
	python3 scripts/run_campaign.py --kind "$(CAMPAIGN_KIND)" --output-root "$(CAMPAIGN_ROOT)" \
		$(if $(strip $(CAMPAIGN_ID)),--campaign-id "$(CAMPAIGN_ID)") \
		$(if $(filter 1 yes true,$(CAMPAIGN_NCU)),--with-ncu)

analyze:
	@test "$(words $(FINAL_CAMPAIGNS))" -eq 3 || { \
		echo "FINAL_CAMPAIGNS must contain exactly three campaign IDs" >&2; exit 2; }
	python3 analysis/analyze.py \
		$(foreach id,$(FINAL_CAMPAIGNS),--campaign "$(CAMPAIGN_ROOT)/$(id)") \
		--output "$(ANALYSIS_OUT)"

profile: build
	@test -n "$(CAMPAIGN_DIR)" || { echo "set CAMPAIGN_DIR=runs/<campaign-id>" >&2; exit 2; }
	python3 scripts/ncu_capture.py --campaign-dir "$(CAMPAIGN_DIR)"

ncu-check: build
	@mkdir -p "$(SMOKE_DIR)"
	python3 scripts/ncu_capture.py --campaign-dir "$(SMOKE_DIR)" \
		--case 00_ldgsts_s2_bif16 --require-complete
