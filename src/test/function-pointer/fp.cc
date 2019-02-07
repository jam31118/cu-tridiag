#include <iostream>
#include <complex>

//template <typename T>
//struct vv {
//  typedef std::complex<double> (*p_comb_t)(T *, T *);
//};

template <typename T>
using p_comb_t = std::complex<double> (*)(T *, T *);

std::complex<double> comb_double(double *real, double *imag) {
  return std::complex<double>(*real + 1, *imag + 1);
}

std::complex<double> comb_int(int *real, int *imag) {
  return std::complex<double>(*real + 1, *imag + 1);
}

//std::complex<double> eval(int real, int imag, std::complex<double> (*p_comb)(void *, void *)) {
//std::complex<double> eval(double *real, double *imag, vv<double>::p_comb_t p_comb) {

//std::complex<double> eval(double *real, double *imag, p_comb_t<double> p_comb) {
//  return p_comb(real, imag);
//}

template <typename T>
std::complex<double> eval_tem(T *real, T *imag, p_comb_t<T> p_comb) {
  return p_comb(real, imag);
}

int main() {
  double x, y;
  int xi = 1, yi = -2;
  std::complex<double> z1, z2;

  x = 1; y = -2;

//  template <> std::complex<double> eval_tem<double>(double )

//  std::complex<double> (*p_comb)(double *, double *);
//  p_comb = comb_double;

//  z = p_comb(&x,&y);

//  z = eval(&x, &y, &comb_double);
  z1 = eval_tem<double>(&x, &y, &comb_double);
  z2 = eval_tem<int>(&xi, &yi, &comb_int);


//  std::cout << "z: " << z << std::endl;
  std::cout << "z1: " << z1 << std::endl;
  std::cout << "z2: " << z2 << std::endl;

  
  

  return 0;

}
