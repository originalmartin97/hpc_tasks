#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number = 0; // Initialize in each process
    int recv_value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

        number = rank + 1;
    MPI_Reduce(&number, &recv_value, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total sum is: %d.\n", recv_value); // Print the reduced value
    }

    printf("Process %d: My number is: %d\n", rank, number); // Print local number

    MPI_Finalize();
    return 0;
}
