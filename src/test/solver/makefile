CUDA_COMPILER = nvcc

CUDA_HOME = /usr/local/cuda
CUDA_LIB_PATH = $(CUDA_HOME)/lib64

CUDA_LIB_PATH_FLAGS = -L$(CUDA_LIB_PATH)
CUDA_LIB_FLAGS = -lcudart -lcusparse
CUDA_INCLUDE_FLAGS = -I$(CUDA_HOME)/include
INCLUDE_FLAGS = -I../../include -I../../../dep/matrix/include

LIB_INSTALL_PATH = ../../../lib

target = tridiag
all: $(target)

#$(target).o: $(target).cu
#	$(CUDA_COMPILER) -c $< $(CUDA_INCLUDE_FLAGS) $(INCLUDE_FLAGS)

#$(target): $(target).o
#	$(CXX) -o $@ $^ $(CUDA_LIB_PATH_FLAGS) $(CUDA_LIB_FLAGS) 

$(target): $(target).cc $(LIB_INSTALL_PATH)/libcu_propagator.so
	$(CXX) $< -o $@ $(CUDA_LIB_FLAGS) $(CUDA_LIB_PATH_FLAGS) $(INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS) -L$(LIB_INSTALL_PATH) -lcu_propagator

$(LIB_INSTALL_PATH)/libcu_propagator.so: $(LIB_INSTALL_PATH)/libcu_propagator.so
	$(MAKE) install -C ../../object

clean:
	$(RM) -f $(target) *.o
	$(MAKE) clean -C ../../object


