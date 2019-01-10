#ifndef _TRIDIAG_KERNEL_H_
#define _TRIDIAG_KERNEL_H_

//// CUDA headers (standard)
#include "cuda_runtime.h"
#include "cusparse_v2.h"

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

template <typename T>
__global__ void tridiag_forward_complex(int m, T *d_ld, T *d_d, T *d_ud, T *d_x_aug, T *d_b_aug) {
  //// NOTES
  // `d_x_aug` and `d_b_aug` are both device pointers to arrays augmented by 2, thus with length `m+2`
  //// NOTES END
  T *d_x = d_x_aug + 1, *d_b = d_b_aug + 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T temp;
  while (tid < m) {
    temp = cuCmul(d_ld[tid], d_x[tid-1]);
    temp = cuCadd(temp, cuCmul(d_d[tid], d_x[tid]));
    temp = cuCadd(temp, cuCmul(d_ud[tid], d_x[tid+1]));
    d_b[tid] = temp;
    tid += blockDim.x * gridDim.x;
  }
}

//// wrapper for dealing with multiple tridiag at once should be implemented ..
//// may be the routine for single tridiag should be defined as __device__ not __global__
//// - but, like `cusparse<t>gtsv2StridedBatch()`, all tridiagonal can be combined into a single 1D array.
//// - I can make use of `diag_unitary_stack` arrays, and wavefunction array, which is also an single 1D array.

#endif // _TRIDIAG_KERNEL_H_
