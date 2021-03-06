// Standard headrs
#include <iostream>
#include <cstdio>
#include <complex>

//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

//// Some home-made headers
#include "helper.h"
#include "cu_helper.h"
#include "cusparse_helper.h"
#include "tridiag_kernel.h"

#include "cu_propagator.h"

#include "tridiag-common.hh"


__global__ void extract_tsurff_psi_and_dpsidr (
    int num_of_wf_lm, int index_at_R, int N_rho,
    cuDoubleComplex *d_psi_R_arr_at_t, 
    cuDoubleComplex *d_dpsi_drho_R_arr_at_t, 
    cuDoubleComplex *d_wf_lm_stack, 
    const double two_over_3delta_rho, 
    const double one_over_12delta_rho ) 
{
  
  cuDoubleComplex *d_wf_lm = NULL;
  double temp_real, temp_imag;
  int tid = threadIdx.x + blockIdx.x * blockDim.x; // same as lm index

  while (tid < num_of_wf_lm) {
    // Determine address of `wf_lm` among the stack of `wf_lm`
    d_wf_lm = d_wf_lm_stack + tid * N_rho;

    d_psi_R_arr_at_t[tid] = d_wf_lm[index_at_R];
    
    temp_real = 
      two_over_3delta_rho * (d_wf_lm[index_at_R+1].x - d_wf_lm[index_at_R-1].x) 
      - one_over_12delta_rho * (d_wf_lm[index_at_R+2].x - d_wf_lm[index_at_R-2].x);
    temp_imag = 
      two_over_3delta_rho * (d_wf_lm[index_at_R+1].y - d_wf_lm[index_at_R-1].y) 
      - one_over_12delta_rho * (d_wf_lm[index_at_R+2].y - d_wf_lm[index_at_R-2].y);
    d_dpsi_drho_R_arr_at_t[tid] = make_cuDoubleComplex(temp_real, temp_imag);
      //= cuCadd( cuCmul(two_over_3delta_rho,cuCadd(d_wf_lm[index_at_R+1], - d_wf_lm[index_at_R-1])), - cuCmul(one_over_12delta_rho, cuCadd(d_wf_lm[index_at_R+2], - d_wf_lm[index_at_R-2])) );

    tid += blockDim.x * gridDim.x;
  }

}




int cu_crank_nicolson_with_tsurff (
    int index_at_R, double delta_rho, int time_index_start, int num_of_time_steps,
    std::complex<double> *h_x_aug, int batch_count,
    std::complex<double> *psi_R_arr, std::complex<double> *dpsi_drho_R_arr, int N, 
    std::complex<double> *h_tridiags_forward[], std::complex<double> *h_tridiags_backward[],
    int num_of_steps_to_print_progress, int rank,
    int block_dim3_in[], int grid_dim3_in[], int batch_stride)
{
  //// Arguments list
  // std::complex<double> *h_tridiags_forward[3], *h_tridiags_backward[3];
  // : in order of `h_ld`, `h_d`, `h_ud`
  // int N;
  // std::complex<double> *h_x_aug;
  // int time_index_start, time_index_max;
  // dim3 blocks, grids;
  
  //// Check argument(s)
  if (batch_stride < 0) { batch_stride = N; } // fall back to default behavior
  

  //// Prepare dim3 variables
  dim3 block_dim3(block_dim3_in[0], block_dim3_in[1], block_dim3_in[2]);
  dim3 grid_dim3(grid_dim3_in[0], grid_dim3_in[1], grid_dim3_in[2]);


  //// Ready for using `cuSPARSE`
  // Create `cuSPARSE` context
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  // Initialize status variables
  cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

  //// Declare device pointers to tridiags arrays
  const int num_of_arrays_in_tridiags = 3;
  std::complex<double> 
    *d_tridiags_forward[num_of_arrays_in_tridiags],
    *d_tridiags_backward[num_of_arrays_in_tridiags],
    *d_x_aug=NULL, *d_x=NULL, *d_b=NULL, *d_b_aug=NULL;

  //// Declare some variables
  int i;

  //// Define some useful variables
  size_t size_of_element = sizeof(std::complex<double>);
  int num_of_elements_in_arr = N * batch_count;
  size_t size_of_arr_in_bytes = num_of_elements_in_arr * size_of_element;
  size_t size_of_aug_arr_in_bytes = (num_of_elements_in_arr + 2) * size_of_element;

//  int batch_index, first_index_at_this_batch, upper_bound_of_index_at_this_batch;
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
  d_x = d_x_aug + 1;



  //// Allocate buffer for `cusparse<T>gtsv2()` routine
  // Get bufer size
  size_t buf_size_for_gtsv2;
  cusparse_status = cusparseZgtsv2StridedBatch_bufferSizeExt(
      handle, N, 
      (cuDoubleComplex *) d_tridiags_forward[i_ud], 
      (cuDoubleComplex *) d_tridiags_forward[i_d], 
      (cuDoubleComplex *) d_tridiags_forward[i_ud], 
      (cuDoubleComplex *) d_b, 
      batch_count, batch_stride, 
      &buf_size_for_gtsv2 );
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }
  fprintf(stdout, "[ LOG ] buf_size_for_gtsv2 = %lu\n", buf_size_for_gtsv2);
  // Allocation
  void *d_buf_for_gtsv2 = NULL;
  cu_err_check( cudaMalloc(&d_buf_for_gtsv2, buf_size_for_gtsv2) );



  // tsurff quantity evaluation related variables
  const int num_of_wf_lm = batch_count;
  const double one_over_12delta_rho = 1.0/(12.0*delta_rho);
  const double two_over_3delta_rho = 2.0/(3.0*delta_rho);

  
  //// Allocate buffer for tsurff quantity evaluation
  std::complex<double> *d_psi_R_arr, *d_dpsi_drho_R_arr, *d_wf_lm_stack;
  std::complex<double> *d_psi_R_arr_at_t, *d_dpsi_drho_R_arr_at_t;
  int tsurff_buffer_length = num_of_wf_lm * num_of_time_steps;
  size_t tsurff_buffer_size = tsurff_buffer_length * sizeof(std::complex<double>);
  cu_err_check( cudaMalloc(&d_psi_R_arr, tsurff_buffer_size) );
  cu_err_check( cudaMalloc(&d_dpsi_drho_R_arr, tsurff_buffer_size) );



  //// Allocate for intermediate result
//  std::complex<double> *h_b = (std::complex<double> *) malloc(size_of_arr_in_bytes);


  //// Start time iteration
  int time_index, time_index_from_zero;
//  int num_of_time_steps = time_index_max - time_index_start;
  int time_index_max = time_index_start + num_of_time_steps;
  int num_of_steps_done_so_far;
//  int rank = 0;
  for (time_index=time_index_start; time_index<time_index_max; ++time_index) {

    time_index_from_zero = time_index - time_index_start;

    //// tsurff quantity evaluation
    d_psi_R_arr_at_t = d_psi_R_arr + time_index_from_zero * num_of_wf_lm;
    d_dpsi_drho_R_arr_at_t = d_dpsi_drho_R_arr + time_index_from_zero * num_of_wf_lm;
    d_wf_lm_stack = d_x;
    
    extract_tsurff_psi_and_dpsidr<<<grid_dim3, block_dim3>>>(
      num_of_wf_lm, index_at_R, batch_stride, 
      (cuDoubleComplex *) d_psi_R_arr_at_t, 
      (cuDoubleComplex *) d_dpsi_drho_R_arr_at_t, 
      (cuDoubleComplex *) d_wf_lm_stack, 
      two_over_3delta_rho, 
      one_over_12delta_rho );



    //// Run forward tridiagonal multiplication on device
    tridiag_forward_complex<<<grid_dim3, block_dim3>>>(
        num_of_elements_in_arr, 
        (cuDoubleComplex *)d_tridiags_forward[i_ld], 
        (cuDoubleComplex *)d_tridiags_forward[i_d], 
        (cuDoubleComplex *)d_tridiags_forward[i_ud], 
        (cuDoubleComplex *)d_x_aug, 
        (cuDoubleComplex *)d_b_aug );
  

    //// Print arrays after forward before backward
//    // copy the intermediate data
//    cu_err_check( cudaMemcpy(h_b, d_b, size_of_arr_in_bytes, cudaMemcpyDeviceToHost) );
//    // print
//    std::cout << "h_b (between): \n";
//    for (batch_index=0; batch_index<batch_count; ++batch_index) {
//      first_index_at_this_batch = batch_index * N;
//      upper_bound_of_index_at_this_batch = (batch_index+1) * N;
//      for (i=first_index_at_this_batch; i<upper_bound_of_index_at_this_batch; ++i) {
//        std::cout << h_b[i] << " ";
//      } std::cout << std::endl;
//    } std::cout << std::endl;
  

    //// Run tridiagonal solve routine
    cusparse_status = cusparseZgtsv2StridedBatch (
        handle, N, 
        (cuDoubleComplex *) d_tridiags_backward[i_ld], 
        (cuDoubleComplex *) d_tridiags_backward[i_d], 
        (cuDoubleComplex *) d_tridiags_backward[i_ud], 
        (cuDoubleComplex *) d_b, 
        batch_count, batch_stride, 
        d_buf_for_gtsv2 ); 
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


    //// Logging
    num_of_steps_to_print_progress = 200;
    if (((time_index + 1) % num_of_steps_to_print_progress) == 0) {
      num_of_steps_done_so_far = time_index - time_index_start + 1;
      if (rank == 0) {
        fprintf(stdout, "[@rank=%d][ LOG ] num_of_steps_done_so_far = %d / %d\n", 
            rank, num_of_steps_done_so_far, num_of_time_steps);
      }
    }


  }

  //// Free intermediate result array;
//  free(h_b);
  //// Free buf for `gtsv2()` routine
  cudaFree(d_buf_for_gtsv2);

  //// Copy data from device to host
  // [NOTE] Since the arrays associated to the pointers in `pd` are only tridiagonals of this linear system's matrix, they don't require memory copy.
  std::complex<double> *h_x = h_x_aug + 1; // temporary variable just for pointing the second element of `h_x_aug` array
  cu_err_check( cudaMemcpy(h_x, d_x, size_of_arr_in_bytes, cudaMemcpyDeviceToHost) );
  // copy tsurff quantities
  cu_err_check( cudaMemcpy(psi_R_arr, d_psi_R_arr, tsurff_buffer_size, cudaMemcpyDeviceToHost) );
  cu_err_check( cudaMemcpy(dpsi_drho_R_arr, d_dpsi_drho_R_arr, tsurff_buffer_size, cudaMemcpyDeviceToHost) );

  //// Free allocated memory
  for (i=0; i<num_of_arrays_in_tridiags; ++i) {
    cudaFree(d_tridiags_forward[i]);
    cudaFree(d_tridiags_backward[i]);
  }
  cudaFree(d_x_aug);
  cudaFree(d_b_aug); // [NOTE] No need for `d_b` since it is just a pointer, pointing to the first element of the array pointed by `d_b_aug`
  cudaFree(d_psi_R_arr);
  cudaFree(d_dpsi_drho_R_arr);


  //// Reset device
  cudaDeviceReset();


  //// Destroy context
  cusparse_status = cusparseDestroy(handle);
  if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
    return cusparse_error_msg(cusparse_status);
  }

  return 0;
}
