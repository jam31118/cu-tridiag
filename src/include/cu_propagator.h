#ifndef _CU_PROPAGATOR_H_
#define _CU_PROPAGATOR_H_

#include <complex>

int tridiag_forward_backward (
    int N, std::complex<double> *h_tridiags_forward[], std::complex<double> *h_tridiags_backward[],
    std::complex<double> *h_x_aug, int time_index_start, int time_index_max,
    int block_dim3_in[], int grid_dim3_in[],
    int batch_count = 1, int batch_stride = -1);


#endif // _CU_PROPAGATOR_H_
