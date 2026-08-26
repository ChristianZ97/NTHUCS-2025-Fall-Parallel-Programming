CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native

TARGET = mpi_odd_even_sort
SOURCE = src/mpi_odd_even_sort.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCE) $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(TARGET)
