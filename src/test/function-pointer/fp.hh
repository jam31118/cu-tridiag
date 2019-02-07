#ifndef _FP_HH_
#define _FP_HH_


#include <complex>


template <typename T>
using p_comb_t = std::complex<double> (*)(T *, T *);

template <typename T>
std::complex<double> eval_tem(T *real, T *imag, p_comb_t<T> p_comb) {
  return p_comb(real, imag);
}


std::complex<double> comb_double(double *real, double *imag);

std::complex<double> comb_int(int *real, int *imag);


#endif // _FP_HH_
