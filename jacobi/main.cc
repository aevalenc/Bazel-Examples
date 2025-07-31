/*
 * Main program for testing Jacobi iteration
 */

#include "jacobi/jacobi.h"
#include <cstdint>
#include <iostream>

int main()
{

  std::cout << "\nStarting Jacobi iteration main...\n";

  // Example matrix and vectors for Jacobi iteration
  const std::int32_t n = 7;

  // clang-format off
  double A[n * n] = { 2, -1,  0,  0,  0,  0,  0,
                     -1,  2, -1,  0,  0,  0,  0,
                      0, -1,  2, -1,  0,  0,  0,
                      0,  0, -1,  2, -1,  0,  0,
                      0,  0,  0, -1,  2, -1,  0,
                      0,  0,  0,  0, -1,  2, -1,
                      0,  0,  0,  0,  0, -1,  2 };
  // clang-format on

  double b[n] = {200, 0, 0, 0, 0, 0, 400};
  double x[n] = {0, 0, 0, 0, 0, 0, 0};

  const std::int32_t max_iter = 1000;
  const double tolerance = 1e-3;

  JacobiIteration(A, b, x, n, max_iter, tolerance);

  return 0;
}
