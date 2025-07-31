/*
 * Jacobi iteration implementation
 */

#include "jacobi/jacobi.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
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
  const auto local_x = new double[n];

  // Send submatrix and vector to each process
  MPI_Scatterv(A, sendcounts_A, displacements_A, MPI_DOUBLE, local_A,
               (end_row - start_row) * n, MPI_DOUBLE, ROOT_PROCESS_LABEL,
               MPI_COMM_WORLD);
  MPI_Scatterv(b, sendcounts_b, displacements_b, MPI_DOUBLE, local_b,
               (end_row - start_row), MPI_DOUBLE, ROOT_PROCESS_LABEL,
               MPI_COMM_WORLD);
  MPI_Bcast(x, n, MPI_DOUBLE, ROOT_PROCESS_LABEL, MPI_COMM_WORLD);
  std::memcpy(local_x, x, n * sizeof(double));

  // Main Jacobi iteration loop
  std::int32_t global_index{0};
  bool converged{false};
  for (std::int32_t iter = 0; iter < max_iter; ++iter)
  {
    double local_sum = 0.0;
    for (std::int32_t i = 0; i < rows_for_this_process; ++i)
    {
      double sum = local_b[i];
      global_index = start_row + i;
      for (std::int32_t j = 0; j < n; ++j)
      {
        if (j != global_index)
        {
          sum -= local_A[i * n + j] * local_x[j];
        }
      }
      x[global_index] = sum / local_A[i * n + global_index];
      local_sum += std::abs(local_x[global_index] - x[global_index]);
    }

    // Gather results from all processes
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // Check convergence
    if (global_sum < tolerance && world_rank == ROOT_PROCESS_LABEL)
    {
      std::cout << "Process " << world_rank << " converged after " << iter + 1
                << " iterations." << std::endl;
      std::memcpy(x, local_x, n * sizeof(double));
      converged = true;
    }
    else if (world_rank == ROOT_PROCESS_LABEL)
    {
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Iteration: " << std::setw(4) << iter + 1
                << " | Global Sum Residual: " << global_sum << "\n";
    }
    MPI_Bcast(&converged, 1, MPI_C_BOOL, ROOT_PROCESS_LABEL, MPI_COMM_WORLD);
    if (converged)
    {
      break;
    }

    // Share the updated solution vector
    MPI_Allgatherv(x + start_row, rows_for_this_process, MPI_DOUBLE, x,
                   sendcounts_b, displacements_b, MPI_DOUBLE, MPI_COMM_WORLD);

    // Update local solution vector
    std::memcpy(local_x, x, n * sizeof(double));
  }

  // Clean up
  delete[] local_A;
  delete[] local_b;
  delete[] local_x;

  // Finalize MPI
  MPI_Finalize();

  return 0.0;
}
