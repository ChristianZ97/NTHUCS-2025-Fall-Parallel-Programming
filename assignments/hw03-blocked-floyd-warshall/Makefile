.DEFAULT_GOAL := all

CXX   ?= g++
NVCC  ?= nvcc
HIPCC ?= hipcc

CXXFLAGS  ?= -O2 -pthread -fopenmp
NVCCFLAGS ?= -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared
HIPFLAGS  ?= -std=c++11 -O3 --offload-arch=gfx908
LDLIBS    ?= -lm

SRC_DIR   := src
BUILD_DIR := build

TARGETS := \
	$(BUILD_DIR)/hw3-1 \
	$(BUILD_DIR)/hw3-2 \
	$(BUILD_DIR)/hw3-2-amd \
	$(BUILD_DIR)/hw3-3 \
	$(BUILD_DIR)/hw3-3-amd

.PHONY: all clean hw3-1 hw3-2 hw3-2-amd hw3-3 hw3-3-amd

all: $(TARGETS)

hw3-1: $(BUILD_DIR)/hw3-1
hw3-2: $(BUILD_DIR)/hw3-2
hw3-2-amd: $(BUILD_DIR)/hw3-2-amd
hw3-3: $(BUILD_DIR)/hw3-3
hw3-3-amd: $(BUILD_DIR)/hw3-3-amd

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/hw3-1: $(SRC_DIR)/hw3-1.cc $(SRC_DIR)/io_utils.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(BUILD_DIR)/hw3-2: $(SRC_DIR)/hw3-2.cu $(SRC_DIR)/io_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

$(BUILD_DIR)/hw3-2-amd: $(SRC_DIR)/hw3-2.hip $(SRC_DIR)/io_utils.h | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $< -o $@ $(LDLIBS)

$(BUILD_DIR)/hw3-3: $(SRC_DIR)/hw3-3.cu $(SRC_DIR)/io_utils.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler="-fopenmp" $< -o $@ $(LDLIBS)

$(BUILD_DIR)/hw3-3-amd: $(SRC_DIR)/hw3-3.hip $(SRC_DIR)/io_utils.h | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) -fopenmp $< -o $@ $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
