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

//// Define macro function
#define MIN(x,y) ((x<y)?x:y)

//// Define my type
typedef double m_t;

//// enum
enum index_name { 
  i_ld, // an index for lower offdiagonal array
  i_ud, // an index for upper offdiagonal array
  i_d, // an index for diagonal array
//  i_x // an index for x array (SHOULD BE DEPRECIATED)
};

template <typename T>
__global__ void tridiag_forward(int m, T *d_ld, T *d_d, T *d_ud, T *d_x_aug, T *d_b_aug) {
  //// NOTES
  // `d_x_aug` : device pointer to an array augmented by 2, thus with length `m+2`
  //// NOTES END
  T *d_x = d_x_aug + 1, *d_b = d_b_aug + 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
//  int i = tid + 1; // this may be combined to the line above
  while (tid < m) {
    d_b[tid] = d_ld[tid] * d_x[tid-1] + d_d[tid] * d_x[tid] + d_ud[tid] * d_x[tid+1];
    tid += blockDim.x * gridDim.x;
  }
}

//// wrapper for dealing with multiple tridiag at once should be implemented ..
//// may be the routine for single tridiag should be defined as __device__ not __global__


void swap_pointers(m_t **p1, m_t **p2) {
  m_t *temp_p = *p1;
  *p1 = *p2;
  *p2 = temp_p;
}

//// main program start
int main(int argc, char *argv[]) {
  
  //// Create `cuSPARSE` context
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  
  //// Initialize status variables
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

  //// Initialize event variables
  cudaEvent_t start, stop;
  cu_err_check( cudaEventCreate(&start) );
  cu_err_check( cudaEventCreate(&stop) );
  float elapsed_time_forward, elapsed_time_backward;


  long i;
  long N = 11;
  size_t
    size_of_m_t = sizeof(m_t),
    size_of_arr = N*size_of_m_t,
    size_of_augmented_arr = (N+2) * size_of_m_t;
  long num_of_aug_arrays = 3;
//  long num_of_aug_arrays = 4;
  m_t
    *tmp=NULL, 
    *ph[num_of_aug_arrays], *h_x_aug,
    *pd[num_of_aug_arrays];
  for (i=0; i<num_of_aug_arrays; ++i) {
    tmp = (m_t *) malloc(size_of_arr);
    if (tmp != NULL) { ph[i] = tmp; }
    else { fprintf(stderr,"[ERROR] during `malloc()`"); return -1; }
  }
  h_x_aug = (m_t *) malloc(size_of_augmented_arr);
  if (h_x_aug == NULL) {fprintf(stderr,"[ERROR] during `malloc()`"); return -1;}
//  ph[i_x] = h_x_aug + 1;
  
  //// Define handles for host arrays
  m_t 
    *h_ld=ph[i_ld], *h_d=ph[i_d], *h_ud=ph[i_ud], *h_x=h_x_aug+1; 
// *h_x=ph[i_x]; // (this is false) ph[i_x] is an augmented vector with length of `N+2`

  //// Fill out tridiagonals and `b` vector (associated with `h_x` array)
  for (i=0; i<N; ++i) {
    h_ld[i] = 1.0; h_d[i] = 1.0; h_ud[i] = 0.0;
    h_x[i] = 0.1 * i;
  }
  h_ld[0] = 0.0; h_ud[N-1] = 0.0;
  h_x[-1] = 0.0; h_x[N] = 0.0;

  //// Print arrays before the calculation
  std::cout << "h_x (before): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;

  ph[i_ld] = h_ld; ph[i_d] = h_d;
  ph[i_ud] = h_ud; // ph[i_x] = h_x;
  
  //// Allocate device memory and copy contents from the host
  for (i=0; i<num_of_aug_arrays; ++i) {
    cu_err_check( cudaMalloc(&pd[i], N*size_of_m_t) );
    cu_err_check ( cudaMemcpy(pd[i], ph[i], N*size_of_m_t, cudaMemcpyHostToDevice) );
  }
  m_t *d_x;
  m_t *d_x_aug = NULL;
  cu_err_check( cudaMalloc(&d_x_aug, size_of_augmented_arr) );
  cu_err_check( cudaMemcpy(d_x_aug, h_x_aug, size_of_augmented_arr, cudaMemcpyHostToDevice) );

  m_t *d_b_aug = NULL;
  cu_err_check( cudaMalloc(&d_b_aug, size_of_augmented_arr) );
  cu_err_check( cudaMemset(d_b_aug, 0, size_of_augmented_arr) ); // [OPTIMIZE] it is enough to set just d_b_aug[0] and d_b_aug[N-1] to zero.

//  pd[i_x] = d_x_aug + 1;

  m_t *d_b = NULL;
//  cu_err_check( cudaMalloc(&d_b, size_of_arr) );
  d_b = d_b_aug + 1;

  //// Forward tridiagonal multiplication configuration
  int num_of_thread_per_block = 128;
  int num_of_blocks_max = 32;
  int num_of_blocks = MIN((N+num_of_thread_per_block-1)/num_of_thread_per_block, num_of_blocks_max);
  // logging
  fprintf(stdout, "[ LOG ] Launching kernel `tridiag_forward` with `%d` blocks with `%d` threads each\n",
      num_of_blocks, num_of_thread_per_block);

  //// Allocate for intermediate result
  m_t *h_b = (m_t *) malloc(size_of_arr);

  //// Configure time iteration
  long num_of_time_steps = 2;
  long time_index, time_index_start = 0;
  long time_index_max=time_index_start+num_of_time_steps;

  //// Start time iteration
  for (time_index=time_index_start; time_index<time_index_max; ++time_index) {
    //// Run forward tridiagonal multiplication on device
    // Execution
    cudaEventRecord(start, 0);
    tridiag_forward<<<num_of_blocks, num_of_thread_per_block>>>(N, pd[i_ld], pd[i_d], pd[i_ud], d_x_aug, d_b_aug);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_forward, start, stop);
  
    //// Print arrays after forward before backward
    // copy the intermediate data
    cu_err_check( cudaMemcpy(h_b, d_b, size_of_arr, cudaMemcpyDeviceToHost) );
    // print
    std::cout << "h_b (between): ";
    for (i=0; i<N; ++i) {
      std::cout << h_b[i] << " ";
    } std::cout << std::endl;
  
    //// Run tridiagonal solve routine
    cudaEventRecord(start, 0);
    cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ud], pd[i_d], pd[i_ud], d_b, N); // changed lower diag to zeros (upper diag is used since it is zero vector in this case)
//    cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ld], pd[i_d], pd[i_ud], d_b, N);
    // [NOTE] 
    // through this tridiagonal solve routine, 
    // the `d_b`, which was originally an array for the right-hand side
    // is overwritten to be an array for `x` array, 
    // which is the solution of this tridiagonal linear system
    // Thus, with some additional consideration, swapping pointers between `d_x_aug` and `d_b_aug` seems to be required.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_backward, start, stop);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
      return cusparse_error_msg(cusparse_status);
    }
    
    //// Swap arrays
    swap_pointers(&d_x_aug, &d_b_aug);
    d_x = d_x_aug + 1; d_b = d_b_aug + 1;

    //// Logging
    fprintf(stdout, "[ LOG ][time_index=%ld] elapsed_time_forward: %.3f ms\n", time_index, elapsed_time_forward);
    fprintf(stdout, "[ LOG ][time_index=%ld] elapsed_time_backward: %.3f ms\n", time_index, elapsed_time_backward);
    std::cout << std::endl;
  }

  //// Free intermediate result array;
  free(h_b);


  //// Copy data from device to host
  for (i=0; i<num_of_aug_arrays; ++i) {
    cu_err_check( cudaMemcpy(ph[i], pd[i], size_of_arr, cudaMemcpyDeviceToHost) );
  }
  cu_err_check( cudaMemcpy(h_x, d_x, size_of_arr, cudaMemcpyDeviceToHost) );
  

  //// Print result
//  h_x = ph[i_x];
  std::cout << "h_x (after): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;

  //// Event log
  fprintf(stdout, "[ LOG ] elapsed_time_forward: %.3f ms\n",elapsed_time_forward);
  fprintf(stdout, "[ LOG ] elapsed_time_backward: %.3f ms\n",elapsed_time_backward);

  //// Destroy events
  cu_err_check( cudaEventDestroy(start) );
  cu_err_check( cudaEventDestroy(stop) );
  
  //// Free allocated memory
  for (i=0; i<num_of_aug_arrays; ++i) {
    free(ph[i]);
    cudaFree(pd[i]);
  }
  free(h_x_aug);
  cudaFree(d_x_aug);

//  cudaFree(d_b);
  cudaFree(d_b_aug);

  //// Reset device
  cudaDeviceReset();

  //// End program
  return 0;
}
