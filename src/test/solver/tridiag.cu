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

//// extra home-made headers
#include "helper.h"

//// Define macro function
#define MIN(x,y) ((x<y)?x:y)

//// Define my type
typedef double m_t;

//// enum
enum index_name { 
  i_ld, // an index for lower offdiagonal array
  i_ud, // an index for upper offdiagonal array
  i_d, // an index for diagonal array
};

template <typename T>
__global__ void tridiag_forward(int m, T *d_ld, T *d_d, T *d_ud, T *d_x_aug, T *d_b_aug) {
  //// NOTES
  // `d_x_aug` and `d_b_aug` are both device pointers to arrays augmented by 2, thus with length `m+2`
  //// NOTES END
  T *d_x = d_x_aug + 1, *d_b = d_b_aug + 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < m) {
    d_b[tid] = d_ld[tid] * d_x[tid-1] + d_d[tid] * d_x[tid] + d_ud[tid] * d_x[tid+1];
    tid += blockDim.x * gridDim.x;
  }
}

//// wrapper for dealing with multiple tridiag at once should be implemented ..
//// may be the routine for single tridiag should be defined as __device__ not __global__
//// - but, like `cusparse<t>gtsv2StridedBatch()`, all tridiagonal can be combined into a single 1D array.
//// - I can make use of `diag_unitary_stack` arrays, and wavefunction array, which is also an single 1D array.


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

  //// Configuration
  long N = 11;
  // forward tridiagonal multiplication configuration
  int num_of_thread_per_block = 128;
  int num_of_blocks_max = 32;
  int num_of_blocks = MIN((N+num_of_thread_per_block-1)/num_of_thread_per_block, num_of_blocks_max);
  // time iteration configuration
  long num_of_time_steps = 3;
  long time_index, time_index_start = 0;
  long time_index_max=time_index_start+num_of_time_steps;

  //// logging for configuration
  fprintf(stdout, "[ LOG ] Launching kernel `tridiag_forward` with `%d` blocks with `%d` threads each\n",
      num_of_blocks, num_of_thread_per_block);
  
  //// Variable declaration with initialization if needed
  long i;
  size_t
    size_of_m_t = sizeof(m_t),
    size_of_arr = N*size_of_m_t,
    size_of_augmented_arr = (N+2) * size_of_m_t;
  long num_of_aug_arrays = 3;
  m_t
    *tmp=NULL, 
    *ph[num_of_aug_arrays], *h_x_aug=NULL, *h_x=NULL, // pointers to host arrays
    *pd[num_of_aug_arrays], *d_x_aug=NULL, *d_x=NULL, *d_b=NULL, *d_b_aug=NULL; // pointers to device arrays

  //// Allocate memory to host arrays
  for (i=0; i<num_of_aug_arrays; ++i) {
    tmp = (m_t *) malloc(size_of_arr);
    if (tmp != NULL) { ph[i] = tmp; }
    else { fprintf(stderr,"[ERROR] during `malloc()`"); return -1; }
  }
  h_x_aug = (m_t *) malloc(size_of_augmented_arr);
  if (h_x_aug == NULL) {fprintf(stderr,"[ERROR] during `malloc()`"); return -1;}
  h_x=h_x_aug+1; 
  
  //// Fill out tridiagonals and `x` vector (associated with `h_x` array)
  for (i=0; i<N; ++i) {
    ph[i_ld][i] = 1.0; ph[i_d][i] = 1.0; ph[i_ud][i] = 0.0;
    h_x[i] = 0.1 * i;
  }
  ph[i_ld][0] = 0.0; ph[i_ud][N-1] = 0.0; // requirement of tridiagonal solver routine from `cuSPARSE`
  h_x_aug[0] = 0.0; h_x_aug[N+1] = 0.0; // fill both ends with zeros

  //// Print arrays before the calculation
  std::cout << "h_x (before): ";
  for (i=0; i<N; ++i) {
    std::cout << h_x[i] << " ";
  } std::cout << std::endl;

  //// Allocate device memory and copy contents from the host
  for (i=0; i<num_of_aug_arrays; ++i) {
    cu_err_check( cudaMalloc(&pd[i], N*size_of_m_t) );
    cu_err_check ( cudaMemcpy(pd[i], ph[i], N*size_of_m_t, cudaMemcpyHostToDevice) );
  }
  cu_err_check( cudaMalloc(&d_x_aug, size_of_augmented_arr) );
  cu_err_check( cudaMemcpy(d_x_aug, h_x_aug, size_of_augmented_arr, cudaMemcpyHostToDevice) );
  cu_err_check( cudaMalloc(&d_b_aug, size_of_augmented_arr) );
  cu_err_check( cudaMemset(d_b_aug, 0, size_of_augmented_arr) ); // [OPTIMIZE] it is enough to set just d_b_aug[0] and d_b_aug[N-1] to zero.
  d_b = d_b_aug + 1;

  //// Allocate buffer for `cusparse<T>gtsv2()` routine
  // Get bufer size
  size_t buf_size_for_gtsv2;
  cusparse_status = cusparseDgtsv2_bufferSizeExt(
      handle, N, 1, pd[i_ud], pd[i_d], pd[i_ud], d_b, N, &buf_size_for_gtsv2);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }
  fprintf(stdout, "[ LOG ] buf_size_for_gtsv2 = %lu\n", buf_size_for_gtsv2);
  // Allocation
  void *d_buf_for_gtsv2 = NULL;
  cu_err_check( cudaMalloc(&d_buf_for_gtsv2, buf_size_for_gtsv2) );

  //// Allocate for intermediate result
  m_t *h_b = (m_t *) malloc(size_of_arr);

  //// Start time iteration
  for (time_index=time_index_start; time_index<time_index_max; ++time_index) {
    //// Run forward tridiagonal multiplication on device
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
    cusparse_status = cusparseDgtsv2(handle, N, 1, pd[i_ud], pd[i_d], pd[i_ud], d_b, N, d_buf_for_gtsv2); // changed lower diag to zeros (upper diag is used since it is zero vector in this case)
//    cusparse_status = cusparseDgtsv(handle, N, 1, pd[i_ud], pd[i_d], pd[i_ud], d_b, N); // changed lower diag to zeros (upper diag is used since it is zero vector in this case)
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
  //// Free buf for `gtsv2()` routine
  cudaFree(d_buf_for_gtsv2);

  //// Copy data from device to host
  // [NOTE] Since the arrays associated to the pointers in `pd` are only tridiagonals of this linear system's matrix, they don't require memory copy.
//  for (i=0; i<num_of_aug_arrays; ++i) {
//    cu_err_check( cudaMemcpy(ph[i], pd[i], size_of_arr, cudaMemcpyDeviceToHost) );
//  }
  cu_err_check( cudaMemcpy(h_x, d_x, size_of_arr, cudaMemcpyDeviceToHost) );
  

  //// Print result
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
  cudaFree(d_b_aug); // [NOTE] No need for `d_b` since it is just a pointer, pointing to the first element of the array pointed by `d_b_aug`

  //// Reset device
  cudaDeviceReset();

  //// End program
  return 0;
}
