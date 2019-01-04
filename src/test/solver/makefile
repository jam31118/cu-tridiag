COMPILER = nvcc

CUDA_HOME = /usr/local/cuda
CUDA_LIB = $(CUDA_HOME)/lib64

CUDA_FLAGS = -lcusparse -I../../include

target = tridiag
all: $(target)

$(target): $(target).cu
	$(COMPILER) -o $@ $^ $(CUDA_FLAGS)

clean:
	$(RM) -f $(target)


