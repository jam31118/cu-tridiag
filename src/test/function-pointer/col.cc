#include "fp.hh"

std::complex<double> comb_double(double *real, double *imag) {
  return std::complex<double>(*real + 1, *imag + 1);
}

std::complex<double> comb_int(int *real, int *imag) {
  return std::complex<double>(*real + 1, *imag + 1);
}

