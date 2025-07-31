/*
 * Jacobi iteration implementation
 */

#ifndef JACOBI_JACOBI_H
#define JACOBI_JACOBI_H

#include <cstdint>

double JacobiIteration(double *A, double *b, double *x, const std::int32_t n,
                       const std::int32_t max_iter, const double tolerance);

#endif // JACOBI_JACOBI_H
