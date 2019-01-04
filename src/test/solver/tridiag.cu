//// C/C++ standard headers
#include <iostream>
#include <cstdlib>
#include <cstdio>

//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// Home-made helper CUDA headers
#include "cusparse_helper.h"
#include "cu_helper.h"


//// Define my type
typedef double m_t;

//// enum
enum index_name { i_ld, i_ud, i_d, i_x };

//// main program start
int main(int argc, char *argv[]) {
  
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  cudaError_t cu_status = cudaSuccess;	

  long i;
  long N = 11;
  size_t size_of_m_t = sizeof(m_t);
  long num_of_aug_arrays = 4;
//  long 
//    num_of_aug_arrays = 4, 
//    num_of_aug_arrays = 4;
//  long total_num_of_arr = num_of_aug_arrays + num_of_aug_arrays;
  m_t *tmp, *ph[num_of_aug_arrays], *pd[num_of_aug_arrays];
  for (i=0; i<num_of_aug_arrays; ++i) {
    tmp = NULL;
    tmp = (m_t *) malloc(N*size_of_m_t);
    if (tmp != NULL) { ph[i] = tmp; }
    else { fprintf(stderr,"[ERROR] during `malloc()`"); return -1; }
  }
  
  //// Define handles for host arrays
  m_t 
    *h_ld=ph[i_ld], *h_d=ph[i_d],
    *h_ud=ph[i_ud], *h_x=ph[i_x]; 

  //// Fill out tridiagonals and `b` vector (associated with `h_x` array)
  for (i=0; i<N; ++i) {
    h_ld[i] = 1.0; h_d[i] = 1.0; h_ud[i] = 0.0;
    h_x[i] = 0.1 * i;
  }
  h_ld[0] = 0.0; h_ud[N-1] = 0.0;

  //// (will be replaced by gpu code) Forward multiplication
  m_t *buf = (m_t *) malloc(N*size_of_m_t);
  buf[0] = h_d[0] * h_x[0] + h_ud[0] * h_x[1];
  for (i=1; i<N-1; ++i) {
    buf[i] = h_ld[i] * h_x[i-1] + h_d[i] * h_x[i] + h_ud[i] * h_x[i+1];
  }
  buf[N-1] = h_ld[N-1] * h_x[N-2] + h_d[N-1] * h_x[N-1];
  memcpy(h_x, buf, N*size_of_m_t);
  free(buf);

  //// Print arrays before the calculation
  std::cout << "h_x (before): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;

  ph[i_ld] = h_ld; ph[i_d] = h_d;
  ph[i_ud] = h_ud; ph[i_x] = h_x;
  
  //// Allocate device memory and copy contents from the host
  for (i=0; i<num_of_aug_arrays; ++i) {
    cu_status = cudaMalloc(&pd[i], N*size_of_m_t);
    if (cu_status != cudaSuccess) { fprintf(stderr, "[ERROR] during `cudaMalloc()` with error: `%s`\n", cudaGetErrorString(cu_status)); return cu_status; }
//    pd[0] = NULL;
    cu_status = cudaMemcpy(pd[i], ph[i], N*size_of_m_t, cudaMemcpyHostToDevice);
    if (cu_status != cudaSuccess) { 
      return cu_error_msg(cu_status);
//      fprintf(stderr, "[ERROR] during `cudaMemcpy()` at file `%s` line `%d`\n", __FILE__, __LINE__); return cu_status; 
    }
  }

  //// Define handles for device arrays
//  m_t 
//    *d_ld=pd[0], *d_d=pd[1], 
//    *d_ud=pd[2], *d_x=pd[3];

  //// Run tridiagonal solve routine
  cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ld], pd[i_d], pd[i_ud], pd[i_x], N);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] during `cusparseDgtsv()` with error: `%s`\n", _cusparseGetErrorEnum(cusparse_status));
    return cusparse_status;
  }

  //// Copy data from device to host
  for (i=0; i<num_of_aug_arrays; ++i) {
    cu_status = cudaMemcpy(ph[i], pd[i], N*size_of_m_t, cudaMemcpyDeviceToHost);
    if (cu_status != cudaSuccess) { fprintf(stderr, "[ERROR] during `cudaMemcpy()`\n"); return cu_status; }
  }

  //// Print result
  h_x = ph[i_x];
  std::cout << "h_x (after): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;
  
  //// Free allocated memory
  for (i=0; i<num_of_aug_arrays; ++i) {
    free(ph[i]);
    cudaFree(pd[i]);
  }

  //// Reset device
  cudaDeviceReset();

  //// End program
  return 0;
}
