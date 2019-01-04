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


#define MIN(x,y) ((x<y)?x:y)


//// Define my type
typedef double m_t;

//// enum
enum index_name { i_ld, i_ud, i_d, i_x };

template <typename T>
__global__ void tridiag_forward(int m, T *d_ld, T *d_d, T *d_ud, T *d_x_aug, T *d_b) {
  //// NOTES
  // `d_x_aug` : device pointer to an array augmented by 2, thus with length `m+2`
  //// NOTES END
  T *d_x = d_x_aug + 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < m) {
    d_b[tid] = d_ld[tid] * d_x[tid-1] + d_d[tid] * d_x[tid] + d_ud[tid] * d_x[tid+1];
    tid += blockDim.x * gridDim.x;
  }
}

//// wrapper for dealing with multiple tridiag at once should be implemented ..
//// may be the routine for single tridiag should be defined as __device__ not __global__


//// main program start
int main(int argc, char *argv[]) {
  
  cusparseHandle_t handle;
  cusparseCreate(&handle);

  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
  cudaError_t cu_status = cudaSuccess;	

  cudaEvent_t start, stop;
  cu_status = cudaEventCreate(&start);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  cu_status = cudaEventCreate(&stop);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }


  long i;
  long N = 101;
  size_t size_of_m_t = sizeof(m_t), size_of_arr;
  long num_of_aug_arrays = 4;
  m_t *tmp, *ph[num_of_aug_arrays], *pd[num_of_aug_arrays];
  for (i=0; i<num_of_aug_arrays; ++i) {
    size_of_arr = N*size_of_m_t;
    if (i==i_x) { 
      continue; 
    }
    tmp = NULL;
    tmp = (m_t *) malloc(size_of_arr);
    if (tmp != NULL) { ph[i] = tmp; }
    else { fprintf(stderr,"[ERROR] during `malloc()`"); return -1; }
  }
  size_t size_of_augmented_arr = (N+2) * size_of_m_t;
  m_t *h_x_aug = (m_t *) malloc(size_of_augmented_arr);
  if (h_x_aug == NULL) {fprintf(stderr,"[ERROR] during `malloc()`"); return -1;}
  ph[i_x] = h_x_aug + 1;
  
  //// Define handles for host arrays
  m_t 
    *h_ld=ph[i_ld], *h_d=ph[i_d],
    *h_ud=ph[i_ud], *h_x=ph[i_x]; // ph[i_x] is an augmented vector with length of `N+2`

  //// Fill out tridiagonals and `b` vector (associated with `h_x` array)
  for (i=0; i<N; ++i) {
    h_ld[i] = 1.0; h_d[i] = 1.0; h_ud[i] = 0.0;
    h_x[i] = 0.1 * i;
  }
  h_ld[0] = 0.0; h_ud[N-1] = 0.0;
  h_x[-1] = 0.0; h_x[N] = 0.0;

  //// (will be replaced by gpu code) Forward multiplication
//  m_t *buf = (m_t *) malloc(N*size_of_m_t);
//  buf[0] = h_d[0] * h_x[0] + h_ud[0] * h_x[1];
//  for (i=1; i<N-1; ++i) {
//    buf[i] = h_ld[i] * h_x[i-1] + h_d[i] * h_x[i] + h_ud[i] * h_x[i+1];
//  }
//  buf[N-1] = h_ld[N-1] * h_x[N-2] + h_d[N-1] * h_x[N-1];
//  memcpy(h_x, buf, N*size_of_m_t);
//  free(buf);

  //// Print arrays before the calculation
//  std::cout << "h_x (before): ";
//  for (i=0; i<N; ++i) {
//    std::cout << h_x[i] << " ";
//  } std::cout << std::endl;

  ph[i_ld] = h_ld; ph[i_d] = h_d;
  ph[i_ud] = h_ud; ph[i_x] = h_x;
  
  //// Allocate device memory and copy contents from the host
  for (i=0; i<num_of_aug_arrays; ++i) {
    if (i==i_x) { continue; }
    cu_status = cudaMalloc(&pd[i], N*size_of_m_t);
    if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
    cu_status = cudaMemcpy(pd[i], ph[i], N*size_of_m_t, cudaMemcpyHostToDevice);
    if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  }
  m_t *d_x_aug = NULL;
  cu_status = cudaMalloc(&d_x_aug, size_of_augmented_arr);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  cu_status = cudaMemcpy(d_x_aug, h_x_aug, size_of_augmented_arr, cudaMemcpyHostToDevice);

  pd[i_x] = d_x_aug + 1;

  m_t *d_b = NULL;
  cu_status = cudaMalloc(&d_b, size_of_arr);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }


  //// Run forward tridiagonal multiplication on device
  int num_of_thread_per_block = 128;
  int num_of_blocks_max = 32;
  int num_of_blocks = MIN((N+num_of_thread_per_block-1)/num_of_thread_per_block, num_of_blocks_max);
  cudaEventRecord(start, 0);
  tridiag_forward<<<num_of_blocks, num_of_thread_per_block>>>(N, pd[i_ld], pd[i_d], pd[i_ud], d_x_aug, d_b);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time_forward, elapsed_time_backward;
  cudaEventElapsedTime(&elapsed_time_forward, start, stop);
  // logging
  fprintf(stdout, "[ LOG ] Launched kernel `tridiag_forward` with `%d` blocks with `%d` threads each\n",
      num_of_blocks, num_of_thread_per_block);

  //// Print arrays after forward before backward
  // copy the intermediate data
  m_t *h_b = (m_t *) malloc(size_of_arr);
  cu_status = cudaMemcpy(h_b, d_b, size_of_arr, cudaMemcpyDeviceToHost);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  free(h_b);
  // print
//  std::cout << "h_b (between): ";
//  for (i=0; i<N; ++i) {
//    std::cout << h_b[i] << " ";
//  } std::cout << std::endl;

  //// Run tridiagonal solve routine
  cudaEventRecord(start, 0);
  cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ld], pd[i_d], pd[i_ud], d_b, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_backward, start, stop);
//  cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ld], pd[i_d], pd[i_ud], pd[i_x], N);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }

  //// Copy data from device to host
  for (i=0; i<num_of_aug_arrays; ++i) {
    if (i==i_x) { continue; }
    cu_status = cudaMemcpy(ph[i], pd[i], size_of_arr, cudaMemcpyDeviceToHost);
    if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  }
  cu_status = cudaMemcpy(h_x, d_b, size_of_arr, cudaMemcpyDeviceToHost);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  

  //// Print result
//  h_x = ph[i_x];
//  std::cout << "h_x (after): ";
//  for (i=0; i<N; ++i) {
//    std::cout << h_x[i] << " ";
//  } std::cout << std::endl;

  //// Event log
  fprintf(stdout, "[ LOG ] elapsed_time_forward: %.3f ms\n",elapsed_time_forward);
  fprintf(stdout, "[ LOG ] elapsed_time_backward: %.3f ms\n",elapsed_time_backward);

  //// Destroy events
  cu_status = cudaEventDestroy(start);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  cu_status = cudaEventDestroy(stop);
  if (cu_status != cudaSuccess) { return cu_error_msg(cu_status); }
  
  //// Free allocated memory
  for (i=0; i<num_of_aug_arrays; ++i) {
    if (i==i_x) { continue; }
    free(ph[i]);
    cudaFree(pd[i]);
  }
  free(h_x_aug);
  cudaFree(d_x_aug);

  cudaFree(d_b);

  //// Reset device
  cudaDeviceReset();

  //// End program
  return 0;
}
