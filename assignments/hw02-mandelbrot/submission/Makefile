CC = gcc
CXX = g++
MPICC = mpicc
MPICXX = mpicxx
LDLIBS = -lpng
BASE_CXXFLAGS = -std=c++17 -O3 -march=native -fopt-info-vec-optimized
PROF_CXXFLAGS = $(BASE_CXXFLAGS) -DPROFILING

TARGETS = hw2a hw2b
PROF_TARGETS = hw2a_prof hw2b_prof

.PHONY: all
all: $(TARGETS)

# hw2a: pthread version
hw2a: hw2a.cc
	$(CXX) $(BASE_CXXFLAGS) -pthread hw2a.cc -o hw2a $(LDLIBS)

# hw2b: MPI+OpenMP version
hw2b: hw2b.cc
	$(MPICXX) $(BASE_CXXFLAGS) -fopenmp hw2b.cc -o hw2b $(LDLIBS)

# Profiling versions
.PHONY: prof
prof: $(PROF_TARGETS)

hw2a_prof: hw2a.cc
	$(CXX) $(PROF_CXXFLAGS) -pthread hw2a.cc -o hw2a_prof $(LDLIBS)

hw2b_prof: hw2b.cc
	$(MPICXX) $(PROF_CXXFLAGS) -fopenmp hw2b.cc -o hw2b_prof $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(PROF_TARGETS)

# List all targets
.PHONY: list
list:
	@echo "hw2 targets: $(PROF_TARGETS)"