/*
 * Jacobi iteration implementation
 */

#include "jacobi/jacobi.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mpi.h>

#define ROOT_PROCESS_LABEL 0

double JacobiIteration(double *A, double *b, double *x, const std::int32_t n,
                       const std::int32_t max_iter, const double tolerance)
{
  // Initialize MPI
  MPI_Init(NULL, NULL);

  // Get the number of processes
  std::int32_t world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  std::int32_t world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Divide the work among processes
  std::int32_t base_rows = n / world_size;
  std::int32_t remainder = n % world_size;
  std::int32_t start_row =
      world_rank * base_rows + std::min(world_rank, remainder);
  std::int32_t rows_for_this_process =
      base_rows + (world_rank < remainder ? 1 : 0);
  std::int32_t end_row = start_row + rows_for_this_process;

  std::int32_t *sendcounts_A = new std::int32_t[world_size];
  std::int32_t *displacements_A = new std::int32_t[world_size];
  std::int32_t *sendcounts_b = new std::int32_t[world_size];
  std::int32_t *displacements_b = new std::int32_t[world_size];
  for (std::int32_t i = 0; i < world_size; ++i)
  {
    std::int32_t rows = base_rows + (i < remainder ? 1 : 0);
    sendcounts_A[i] = rows * n;
    displacements_A[i] = (i * base_rows + std::min(i, remainder)) * n;
    sendcounts_b[i] = rows;
    displacements_b[i] = i * base_rows + std::min(i, remainder);
  }

  // Initialize local buffers for submatrix and vector
  const auto local_A = new double[(end_row - start_row) * n];
  const auto local_b = new double[(end_row - start_row)];
  const auto local_x = new double[(end_row - start_row)];

  // Send submatrix and vector to each process
  MPI_Scatterv(A, sendcounts_A, displacements_A, MPI_DOUBLE, local_A,
               (end_row - start_row) * n, MPI_DOUBLE, ROOT_PROCESS_LABEL,
               MPI_COMM_WORLD);
  MPI_Scatterv(b, sendcounts_b, displacements_b, MPI_DOUBLE, local_b,
               (end_row - start_row), MPI_DOUBLE, ROOT_PROCESS_LABEL,
               MPI_COMM_WORLD);
  MPI_Bcast(x, n, MPI_DOUBLE, ROOT_PROCESS_LABEL, MPI_COMM_WORLD);

  //   std::memcpy(local_A, A + start_row * n,
  //               (end_row - start_row) * n * sizeof(double));
  //   std::memcpy(local_b, b, n * sizeof(double));
  //   std::memcpy(local_x, x, n * sizeof(double));

  for (std::int32_t i = 0; i < (end_row - start_row) * n; ++i)
  {
    std::cout << "Process " << world_rank << " A[" << i << "] = " << local_A[i]
              << "\n";
  }

  // Main Jacobi iteration loop
  //   for (std::int32_t iter = 0; iter < max_iter; ++iter)
  //   {
  //     double local_sum = 0.0;
  //     for (std::int32_t i = start_row; i < end_row; ++i)
  //     {
  //       double sum = local_b[i];
  //       for (std::int32_t j = 0; j < n; ++j)
  //       {
  //         if (j != i)
  //         {
  //           sum -= local_A[i * n + j] * local_x[j];
  //         }
  //       }
  //       local_x[i] = sum / local_A[i * n + i];
  //       local_sum += std::abs(local_x[i] - x[i]);
  //     }

  //     // Gather results from all processes
  //     double global_sum;
  //     MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
  //                   MPI_COMM_WORLD);

  //     // Check convergence
  //     if (global_sum < tolerance)
  //     {
  //       std::cout << "Process " << world_rank << " converged after " << iter
  //       + 1
  //                 << " iterations." << std::endl;
  //       break;
  //     }
  //     else
  //     {
  //       std::cout << "Process " << world_rank << " did not converge after "
  //                 << iter + 1 << " iterations." << std::endl;
  //     }

  //     // Update the global solution vector
  //     MPI_Allgather(local_x + start_row, rows_per_process, MPI_DOUBLE, x,
  //                   rows_per_process, MPI_DOUBLE, MPI_COMM_WORLD);
  //     if (end_row < n)
  //     {
  //       std::fill(x + end_row, x + n, 0.0);
  //     }
  //   }

  // Clean up
  delete[] local_A;
  delete[] local_b;
  delete[] local_x;

  // Finalize MPI
  MPI_Finalize();

  return 0.0;
}
