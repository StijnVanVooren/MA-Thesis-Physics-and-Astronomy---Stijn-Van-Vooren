#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max_driver_1 = 15;
    int max_driver_2 = 15;
    int max_target = 15;

    // Calculate the number of iterations for the two outer loops and each rank
    int num_iterations_c1_c2 = (max_driver_1 * max_driver_2) / size;
    int remainder_c1_c2 = (max_driver_1 * max_driver_2) % size;

    int rank_start_c1_c2 = rank * num_iterations_c1_c2;
    int rank_end_c1_c2 = rank_start_c1_c2 + num_iterations_c1_c2;

    if (rank < remainder_c1_c2) {
        rank_start_c1_c2 += rank;
        rank_end_c1_c2 += rank + 1;
    } else {
        rank_start_c1_c2 += remainder_c1_c2;
        rank_end_c1_c2 += remainder_c1_c2;
    }

    // Perform the loops on each rank
    for (int c1_c2 = rank_start_c1_c2; c1_c2 < rank_end_c1_c2; c1_c2++) {
        // Extract the values of c1 and c2 from the combined index
        int c1 = c1_c2 / max_driver_2;
        int c2 = c1_c2 % max_driver_2;

        for (int c3 = c1; c3 < max_target; c3++) {
            std::cout << c1 << " " << c2 << " " << c3 << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
