#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Function
double f(double x) {
    return 4.0 / (1.0 + x * x);
}

int main( int argc, char **argv ){
    int rank, size, counter = 0;
    double y = 0.0, sum = 0.0, average = 0.0;
    
    // MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);
    double x = (double)rand() / (double)RAND_MAX;
    y = f(x);

    MPI_Reduce(&y, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        average = sum / size;
        printf("(⌐■_■) Average f(xi): %f\n", average);
    }

    MPI_Finalize();
    return 0;
}