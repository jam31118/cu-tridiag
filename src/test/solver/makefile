CUDA_COMPILER = nvcc

CUDA_HOME = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_HOME)/lib64

CUDA_LIB_PATH_FLAGS = -L$(CUDA_LIB_PATH)
CUDA_LIB_FLAGS = -lcudart -lcusparse
CUDA_INCLUDE_FLAGS = -I$(CUDA_HOME)/include
INCLUDE_FLAGS = -I../../include

target = tridiag
all: $(target)

$(target).o: $(target).cu
	$(CUDA_COMPILER) -c $< $(CUDA_INCLUDE_FLAGS) $(INCLUDE_FLAGS)

$(target): $(target).o
	$(CXX) -o $@ $^ $(CUDA_LIB_PATH_FLAGS) $(CUDA_LIB_FLAGS) 

clean:
	$(RM) -f $(target) *.o


