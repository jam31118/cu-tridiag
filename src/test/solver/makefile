include ../../var.mk

# flags for CUDA
CUDA_LIB_PATH_FLAGS = -L$(CUDA_LIB_PATH)
CUDA_LIBS = -lcudart -lcusparse 
CUDA_LIB_FLAGS = $(CUDA_LIB_PATH_FLAGS) $(CUDA_LIBS)

CUDA_INCLUDE_FLAGS = -I$(CUDA_HOME)/include


# flags for `matrix` library
MATRIX_FLAGS = -I$(MATRIX_HOME)/include


# flags for `cu_propagator` library
INCLUDE_FLAGS = -I../../../include 
LIB_INSTALL_PATH = ../../../lib
CU_PROPAGATOR_FLAGS = -L$(LIB_INSTALL_PATH) -lcu_propagator
OBJ_PATH = ../../object



target = tridiag
all: $(target)

$(target): $(target).cc $(LIB_INSTALL_PATH)/libcu_propagator.so
	$(CXX) $< -o $@ $(CUDA_LIB_FLAGS) $(INCLUDE_FLAGS) $(MATRIX_FLAGS) $(CU_PROPAGATOR_FLAGS)

$(LIB_INSTALL_PATH)/libcu_propagator.so: 
	$(MAKE) install -C $(OBJ_PATH)

clean:
	$(RM) -f $(target) *.o
	$(MAKE) clean -C $(OBJ_PATH)

