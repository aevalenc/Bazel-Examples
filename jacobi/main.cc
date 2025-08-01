/*
 * Main program for testing Jacobi iteration
 */

#include "jacobi/jacobi.h"
#include "jacobi/utils.h"
#include <cstdint>
#include <iostream>
#include <time.h>

int main()
{

  std::cout << "\nStarting Jacobi iteration main...\n";

  // Example matrix and vectors for Jacobi iteration
  const std::int32_t n = 10001;

  // clang-format off
  const auto A = InitializeLaplaceMatrix(n);
  // clang-format on

  double *b = new double[n];
  for (std::int32_t i = 0; i < n; ++i)
  {
    if (i == 0)
    {
      b[i] = 200.0;
    }
    else if (i == n - 1)
    {
      b[i] = 400.0;
    }
    else
    {
      b[i] = 0.0;
    }
  }

  double *x = new double[n];
  for (std::int32_t i = 0; i < n; ++i)
  {
    x[i] = 0.0; // Initial guess
  }

  const std::int32_t max_iter = 1000;
  const double tolerance = 1e-3;

  const auto start = clock();
  JacobiIteration(A, b, x, n, max_iter, tolerance);
  const auto end = clock();
  const double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
  std::cout << "Jacobi iteration completed in " << elapsed_time
            << " seconds.\n";

  delete[] A;
  delete[] b;
  delete[] x;

  return 0;
}
