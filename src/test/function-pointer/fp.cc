#include <iostream>
#include <complex>

#include "fp.hh"

int main() {

  double x = 1, y = -2;
  int xi = 1, yi = -2;
  std::complex<double> z1, z2;

  z1 = eval_tem<double>(&x, &y, &comb_double);
  z2 = eval_tem<int>(&xi, &yi, &comb_int);

  std::cout << "z1: " << z1 << std::endl;
  std::cout << "z2: " << z2 << std::endl;

  return 0;

}
