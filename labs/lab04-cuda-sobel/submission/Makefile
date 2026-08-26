NVCC     = nvcc 
NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61

HIPCC = hipcc
HIPCCFLAGS  = -O3 -std=c++11 --offload-arch=gfx90a

LDFLAGS = -lpng -lz

TARGETS = sobel-opt

sobel-opt: sobel-opt.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
