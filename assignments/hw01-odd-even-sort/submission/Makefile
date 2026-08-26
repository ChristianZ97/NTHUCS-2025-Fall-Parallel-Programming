CC = mpicc
CXX = mpicxx
BASE_CXXFLAGS = -std=c++17 -O3 -march=native
PROF_CXXFLAGS = $(BASE_CXXFLAGS) -DPROFILING
TARGETS = hw1
PROF_TARGET = hw1_prof

.PHONY: all
all: $(TARGETS)

hw1: hw1.cc
	$(CXX) $(BASE_CXXFLAGS) hw1.cc -o hw1

.PHONY: prof
prof: $(PROF_TARGET)

$(PROF_TARGET): hw1.cc
	$(CXX) $(PROF_CXXFLAGS) hw1.cc -o $(PROF_TARGET)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(PROF_TARGET)
