CC     = gcc
CXX    = g++
NVCC   = nvcc
HIPCC  = hipcc

CXXFLAGS      = -O2 -pthread -fopenmp
NVFLAGS       = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -cudart shared
HIPCCFLAGS    = -std=c++11 -O3 --offload-arch=gfx908

# Profiling variants (enable -DPROFILING)
PROF_CXXFLAGS   = $(CXXFLAGS) -DPROFILING
PROF_NVFLAGS    = $(NVFLAGS) -DPROFILING
PROF_HIPCCFLAGS = $(HIPCCFLAGS) -DPROFILING

LDFLAGS = -lm
EXES    = hw3-1 hw3-2 hw3-3 hw3-2-amd hw3-3-amd
PROF_EXES = hw3-1-prof hw3-2-prof hw3-3-prof

hw3-3: NVFLAGS += -Xcompiler="-fopenmp"
hw3-3-amd: HIPCCFLAGS += -fopenmp

.PHONY: all clean prof

all: $(EXES)

prof: $(PROF_EXES)

clean:
	rm -f $(EXES) $(PROF_EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

# =========================
# Normal targets
# =========================
hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

# =========================
# Profiling targets (-DPROFILING)
# =========================
hw3-1-prof: hw3-1.cc
	$(CXX) $(PROF_CXXFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-prof: hw3-2.cu
	$(NVCC) $(PROF_NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-prof: hw3-3.cu
	$(NVCC) $(PROF_NVFLAGS) $(LDFLAGS) -o $@ $?
