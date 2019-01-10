
CUDA_COMPILER = nvcc

CUDA_HOME = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_HOME)/lib64

CUDA_LIB_PATH_FLAGS = -L$(CUDA_LIB_PATH)
CUDA_LIB_FLAGS = -lcudart -lcusparse
CUDA_INCLUDE_FLAGS = -I$(CUDA_HOME)/include

INCLUDE_FLAGS = -I../include -I../../dep/matrix/include


LIB_INSTALL_PATH = ../../lib


all: install

install: libcu_propagator.so
	if [ ! -d "$$LIB_INSTALL_PATH" ]; then mkdir -p $(LIB_INSTALL_PATH); mv $^ $(LIB_INSTALL_PATH); fi

libcu_propagator.so: cu_propagator.o gpu_code.o
	$(CXX) -shared $^ -o $@ $(CUDA_LIB_FLAGS) $(CUDA_LIB_PATH_FLAGS) -fPIC

gpu_code.o: cu_propagator.o
	$(CUDA_COMPILER) $^ -o $@ -dlink --compiler-options -fPIC

cu_propagator.o: cu_propagator.cu
	$(CUDA_COMPILER) $< -o $@ -c $(CUDA_INCLUDE_FLAGS) $(INCLUDE_FLAGS) -dc --compiler-options -fPIC


clean:
	$(RM) -f *.o *.so $(LIB_INSTALL_PATH)/*.so

