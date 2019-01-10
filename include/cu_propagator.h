#ifndef _CU_PROPAGATOR_H_
#define _CU_PROPAGATOR_H_

#include <complex>

//int tridiag_forward_backward (
//    int N, std::complex<double> *h_tridiags_forward[], std::complex<double> *h_tridiags_backward[],
//    std::complex<double> *h_x_aug, int time_index_start, int time_index_max,
//    int block_dim3_in[], int grid_dim3_in[],
//    int batch_count = 1, int batch_stride = -1);

int cu_crank_nicolson_with_tsurff (
    int index_at_R, double delta_rho, int time_index_start, int num_of_time_steps,
    std::complex<double> *h_x_aug, int batch_count,
    std::complex<double> *psi_R_arr, std::complex<double> *dpsi_drho_R_arr, int N, 
    std::complex<double> *h_tridiags_forward[], std::complex<double> *h_tridiags_backward[],
    int num_of_steps_to_print_progress, int rank,
    int block_dim3_in[], int grid_dim3_in[], int batch_stride);

#endif // _CU_PROPAGATOR_H_
