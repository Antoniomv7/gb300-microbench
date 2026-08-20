NVCC ?= nvcc
CUOBJDUMP ?= cuobjdump
ARCH ?= sm_103a
BUILD_DIR ?= ../build/memory_paths
SASS_DIR ?= ../build/sass
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo

.PHONY: all sass clean
all: $(BUILD_DIR)/ldgsts $(BUILD_DIR)/tma

$(BUILD_DIR)/ldgsts: ldgsts.cu Makefile ../VERSIONS.env
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $<

$(BUILD_DIR)/tma: tma.cu Makefile ../VERSIONS.env
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -o $@ $<

sass: all
	@mkdir -p $(SASS_DIR)
	$(CUOBJDUMP) --dump-sass $(BUILD_DIR)/ldgsts > $(SASS_DIR)/ldgsts.sass
	$(CUOBJDUMP) --dump-sass $(BUILD_DIR)/tma > $(SASS_DIR)/tma.sass

clean:
	rm -f $(BUILD_DIR)/ldgsts $(BUILD_DIR)/tma
	rm -f $(SASS_DIR)/ldgsts.sass $(SASS_DIR)/tma.sass
