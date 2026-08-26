# ==========================================
# Makefile for C, C++, CUDA, HIP
# Supports: Standard, Debug (-DDEBUG), Profiling (-DPROFILING)
# ==========================================

# Compilers
CC     = gcc
NVCC   = nvcc
HIPCC  = hipcc

# Base Flags (Optimization & Architecture)
CFLAGS_BASE   = -Wall -g
NVFLAGS_BASE  = -std=c++11 -O2 -Xptxas="-v" -arch=sm_61 -fmad=false
HIPCCFLAGS    = -std=c++14 -O2 --offload-arch=gfx908 -ffp-contract=off
LDFLAGS       = -lm

# Variant Flags
# 1. Debug: Adds (symbols) and -DDEBUG (macro)
CFLAGS_DBG    = $(CFLAGS_BASE) -DDEBUG
NVFLAGS_DBG   = $(NVFLAGS_BASE) -DDEBUG -g -G
HIPCCFLAGS_DBG= $(HIPCCFLAGS) -DDEBUG -g

# 2. Profiling: Adds -DPROFILING (macro)
CFLAGS_PROF   = $(CFLAGS_BASE) -DPROFILING
NVFLAGS_PROF  = $(NVFLAGS_BASE) -DPROFILING
HIPCCFLAGS_PROF= $(HIPCCFLAGS) -DPROFILING

# ------------------------------------------
# Source & Target Generation
# ------------------------------------------
C_SRCS    := $(wildcard *.c)
CU_SRCS   := $(wildcard *.cu)
HIP_SRCS  := $(wildcard *.hip)

# Standard Targets
C_EXES    := $(C_SRCS:.c=_c)
CU_EXES   := $(CU_SRCS:.cu=_cu)
HIP_EXES  := $(HIP_SRCS:.hip=_hip)

# Debug Targets
C_DBG     := $(C_SRCS:.c=_c_debug)
CU_DBG    := $(CU_SRCS:.cu=_cu_debug)
HIP_DBG   := $(HIP_SRCS:.hip=_hip_debug)

# Profiling Targets
C_PROF    := $(C_SRCS:.c=_c_prof)
CU_PROF   := $(CU_SRCS:.cu=_cu_prof)
HIP_PROF  := $(HIP_SRCS:.hip=_hip_prof)

# Grouped Lists
TARGETS_ALL  = $(C_EXES) $(CU_EXES) $(HIP_EXES)
TARGETS_DBG  = $(C_DBG) $(CU_DBG) $(HIP_DBG)
TARGETS_PROF = $(C_PROF) $(CU_PROF) $(HIP_PROF)

# ------------------------------------------
# Phony Rules
# ------------------------------------------
.PHONY: all debug prof clean c cuda hip

# Default: Build only Standard versions
all: $(TARGETS_ALL)
	@echo "Build complete (Standard: C/CUDA/HIP)."

# Build Debug versions
debug: $(TARGETS_DBG)
	@echo "Build complete (Debug: C/CUDA/HIP)."

# Build Profiling versions
prof: $(TARGETS_PROF)
	@echo "Build complete (Profiling: C/CUDA/HIP)."

# Individual language targets
c: $(C_EXES)
	@echo "C builds complete."

cuda: $(CU_EXES)
	@echo "CUDA builds complete."

hip: $(HIP_EXES)
	@echo "HIP builds complete."

clean:
	rm -f $(TARGETS_ALL) $(TARGETS_DBG) $(TARGETS_PROF)
	@echo "Clean complete."

# ------------------------------------------
# Compilation Rules
# ------------------------------------------

# --- Standard Rules ---
%_c: %.c
	$(CC) $(CFLAGS_BASE) -o $@ $? $(LDFLAGS)

%_cu: %.cu
	$(NVCC) $(NVFLAGS_BASE) -o $@ $? $(LDFLAGS)

%_hip: %.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ $? $(LDFLAGS)

# --- Debug Rules ---
%_c_debug: %.c
	$(CC) $(CFLAGS_DBG) -o $@ $? $(LDFLAGS)

%_cu_debug: %.cu
	$(NVCC) $(NVFLAGS_DBG) -o $@ $? $(LDFLAGS)

%_hip_debug: %.hip
	$(HIPCC) $(HIPCCFLAGS_DBG) -o $@ $? $(LDFLAGS)

# --- Profiling Rules ---
%_c_prof: %.c
	$(CC) $(CFLAGS_PROF) -o $@ $? $(LDFLAGS)

%_cu_prof: %.cu
	$(NVCC) $(NVFLAGS_PROF) -o $@ $? $(LDFLAGS)

%_hip_prof: %.hip
	$(HIPCC) $(HIPCCFLAGS_PROF) -o $@ $? $(LDFLAGS)
