NVCC = nvcc
HIPCC = hipcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
HIPCCFLAGS = -std=c++14 -O3 --offload-arch=gfx908

TARGETS = hw4 hw4-amd

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

hw4: src/flash_attention_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LDFLAGS)

hw4-amd: src/flash_attention_hip.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ $< $(LDFLAGS)
