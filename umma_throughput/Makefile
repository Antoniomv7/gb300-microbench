NVCC ?= nvcc
CUOBJDUMP ?= cuobjdump
ARCH ?= sm_103a
VIRTUAL_ARCH := compute_$(patsubst sm_%,%,$(ARCH))
BUILD_DIR ?= ../build/umma_throughput
SASS_DIR ?= ../build/sass
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo

.PHONY: all sass clean scaling scaling-sass
all: $(BUILD_DIR)/umma_1sm $(BUILD_DIR)/umma_2sm

# Device scaling is excluded from the default build.
scaling: $(BUILD_DIR)/umma_device_scaling

$(BUILD_DIR)/umma_1sm: umma_1sm.cu Makefile ../VERSIONS.env
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(VIRTUAL_ARCH) -code=$(ARCH) -o $@ $<

$(BUILD_DIR)/umma_2sm: umma_2sm.cu Makefile ../VERSIONS.env
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(VIRTUAL_ARCH) -code=$(ARCH) -o $@ $<

$(BUILD_DIR)/umma_device_scaling: umma_device_scaling.cu Makefile ../VERSIONS.env
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(VIRTUAL_ARCH) -code=$(ARCH) -o $@ $<

sass: all
	@mkdir -p $(SASS_DIR)
	$(CUOBJDUMP) --dump-sass $(BUILD_DIR)/umma_1sm > $(SASS_DIR)/umma_1sm.sass
	$(CUOBJDUMP) --dump-sass $(BUILD_DIR)/umma_2sm > $(SASS_DIR)/umma_2sm.sass

scaling-sass: scaling
	@mkdir -p $(SASS_DIR)
	$(CUOBJDUMP) --dump-sass $(BUILD_DIR)/umma_device_scaling > $(SASS_DIR)/umma_device_scaling.sass

clean:
	rm -f $(BUILD_DIR)/umma_1sm $(BUILD_DIR)/umma_2sm $(BUILD_DIR)/umma_device_scaling
	rm -f $(SASS_DIR)/umma_1sm.sass $(SASS_DIR)/umma_2sm.sass $(SASS_DIR)/umma_device_scaling.sass
