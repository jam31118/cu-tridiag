#ifndef _TRIDIAG_H_
#define _TRIDIAG_H_

#include "helper.h"

//// Define mapping from index_name to integer as indices
enum index_name { 
  i_ld, // an index for lower offdiagonal array
  i_ud, // an index for upper offdiagonal array
  i_d, // an index for diagonal array
};


template <typename m_t>
int tridiag_forward_backward (
    int N, m_t *h_tridiags_forward[], m_t *h_tridiags_backward[],
    m_t *h_x_aug, int time_index_start, int time_index_max,
    dim3 block_dim3, dim3 grid_dim3 )
{
  //// Arguments list
  // m_t *h_tridiags_forward[3], *h_tridiags_backward[3];
  // : in order of `h_ld`, `h_d`, `h_ud`
  // int N;
  // m_t *h_x_aug;
  // int time_index_start, time_index_max;
  // dim3 blocks, grids;

  //// Ready for using `cuSPARSE`
  // Create `cuSPARSE` context
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  // Initialize status variables
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

  //// Declare device pointers to tridiags arrays
  const int num_of_arrays_in_tridiags = 3;
  m_t 
    *d_tridiags_forward[num_of_arrays_in_tridiags],
    *d_tridiags_backward[num_of_arrays_in_tridiags],
    *d_x_aug=NULL, *d_x=NULL, *d_b=NULL, *d_b_aug=NULL;

  //// Declare some variables
  int i;

  //// Define some useful variables
  size_t size_of_element = sizeof(m_t);
  size_t size_of_arr_in_bytes = N * size_of_element;
  size_t size_of_aug_arr_in_bytes = (N + 2) * size_of_element;

  //// Allocate device memory and copy contents from the host
  for (i=0; i<num_of_arrays_in_tridiags; ++i) {
    // `d_tridaigs_forward`
    cu_err_check( cudaMalloc(&d_tridiags_forward[i], size_of_arr_in_bytes) );
    cu_err_check( cudaMemcpy(d_tridiags_forward[i], h_tridiags_forward[i], size_of_arr_in_bytes, cudaMemcpyHostToDevice) );
    // `d_tridaigs_backward`
    cu_err_check( cudaMalloc(&d_tridiags_backward[i], size_of_arr_in_bytes) );
    cu_err_check( cudaMemcpy(d_tridiags_backward[i], h_tridiags_backward[i], size_of_arr_in_bytes, cudaMemcpyHostToDevice) );
  }
  // `d_x_aug`
  cu_err_check( cudaMalloc(&d_x_aug, size_of_aug_arr_in_bytes) );
  cu_err_check( cudaMemcpy(d_x_aug, h_x_aug, size_of_aug_arr_in_bytes, cudaMemcpyHostToDevice) );
  // `d_b_aug`
  cu_err_check( cudaMalloc(&d_b_aug, size_of_aug_arr_in_bytes) );
  cu_err_check( cudaMemset(d_b_aug, 0, size_of_aug_arr_in_bytes) ); // [OPTIMIZE] it is enough to set just d_b_aug[0] and d_b_aug[N-1] to zero.
  // `d_b`
  d_b = d_b_aug + 1;

  //// Allocate buffer for `cusparse<T>gtsv2()` routine
  // Get bufer size
  size_t buf_size_for_gtsv2;
  cusparse_status = cusparseDgtsv2_bufferSizeExt(
      handle, N, 1, 
      d_tridiags_forward[i_ud], 
      d_tridiags_forward[i_d], 
      d_tridiags_forward[i_ud], 
      d_b, N, &buf_size_for_gtsv2 );
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }
  fprintf(stdout, "[ LOG ] buf_size_for_gtsv2 = %lu\n", buf_size_for_gtsv2);
  // Allocation
  void *d_buf_for_gtsv2 = NULL;
  cu_err_check( cudaMalloc(&d_buf_for_gtsv2, buf_size_for_gtsv2) );

  //// Allocate for intermediate result
  m_t *h_b = (m_t *) malloc(size_of_arr_in_bytes);

  //// Start time iteration
  int time_index;
  for (time_index=time_index_start; time_index<time_index_max; ++time_index) {
    //// Run forward tridiagonal multiplication on device
    tridiag_forward<<<grid_dim3, block_dim3>>>(
        N, 
        d_tridiags_forward[i_ld], 
        d_tridiags_forward[i_d], 
        d_tridiags_forward[i_ud], 
        d_x_aug, d_b_aug );
  
    //// Print arrays after forward before backward
    // copy the intermediate data
    cu_err_check( cudaMemcpy(h_b, d_b, size_of_arr_in_bytes, cudaMemcpyDeviceToHost) );
    // print
    std::cout << "h_b (between): ";
    for (i=0; i<N; ++i) {
      std::cout << h_b[i] << " ";
    } std::cout << std::endl;
  
    //// Run tridiagonal solve routine
    cusparse_status = cusparseDgtsv2(
        handle, N, 1, 
        d_tridiags_backward[i_ld], 
        d_tridiags_backward[i_d], 
        d_tridiags_backward[i_ud], 
        d_b, N, d_buf_for_gtsv2 ); 
    // [NOTE] 
    // through this tridiagonal solve routine, 
    // the `d_b`, which was originally an array for the right-hand side
    // is overwritten to be an array for `x` array, 
    // which is the solution of this tridiagonal linear system
    // Thus, with some additional consideration, swapping pointers between `d_x_aug` and `d_b_aug` seems to be required.
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
      return cusparse_error_msg(cusparse_status);
    }
    
    //// Swap arrays
    swap_pointers(&d_x_aug, &d_b_aug);
    d_x = d_x_aug + 1; d_b = d_b_aug + 1;
  }

  //// Free intermediate result array;
  free(h_b);
  //// Free buf for `gtsv2()` routine
  cudaFree(d_buf_for_gtsv2);

  //// Copy data from device to host
  // [NOTE] Since the arrays associated to the pointers in `pd` are only tridiagonals of this linear system's matrix, they don't require memory copy.
  m_t *h_x = h_x_aug + 1; // temporary variable just for pointing the second element of `h_x_aug` array
  cu_err_check( cudaMemcpy(h_x, d_x, size_of_arr_in_bytes, cudaMemcpyDeviceToHost) );

  //// Free allocated memory
  for (i=0; i<num_of_arrays_in_tridiags; ++i) {
    cudaFree(d_tridiags_forward[i]);
    cudaFree(d_tridiags_backward[i]);
  }
  cudaFree(d_x_aug);
  cudaFree(d_b_aug); // [NOTE] No need for `d_b` since it is just a pointer, pointing to the first element of the array pointed by `d_b_aug`

  //// Reset device
  cudaDeviceReset();

  //// Destroy context
  cusparse_status = cusparseDestroy(handle);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }

  return 0;
}


#endif // _TRIDIAG_H_
