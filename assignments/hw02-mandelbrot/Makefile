CXX = g++
MPICXX = mpicxx

CPPFLAGS = -Isrc
CXXFLAGS = -std=c++17 -O3 -march=native
LDLIBS = -lpng

COMMON_SOURCE = src/png_writer.cc
TARGETS = hw2a hw2b

all: $(TARGETS)

hw2a: src/mandelbrot_pthreads.cc $(COMMON_SOURCE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread $^ -o $@ $(LDLIBS)

hw2b: src/mandelbrot_mpi_openmp.cc $(COMMON_SOURCE)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) -fopenmp $^ -o $@ $(LDLIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
