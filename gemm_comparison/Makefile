NVCC ?= nvcc
PYTHON ?= python3
ARCH ?= sm_103a
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
BRIDGE_DIR ?= /tmp/gb300-cublaslt
BRIDGE_LIB := $(BRIDGE_DIR)/libcublaslt_bridge.so
OUTPUT ?= ../runs/gemm_comparison.csv
WARMUP ?= 2
ITERATIONS ?= 10

.PHONY: bridge run
bridge:
	@mkdir -p $(BRIDGE_DIR)
	$(NVCC) -std=c++17 -O3 -lineinfo -Xcompiler -fPIC -shared \
		-arch=$(VIRTUAL_ARCH) -code=$(ARCH) \
		-o $(BRIDGE_LIB) cublaslt_bridge.cu -lcublasLt -lcudart

run: bridge
	@mkdir -p $(dir $(OUTPUT))
	$(PYTHON) gemm_comparison.py --warmup-iterations $(WARMUP) \
		--iterations $(ITERATIONS) > $(OUTPUT)
