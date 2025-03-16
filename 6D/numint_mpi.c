#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define M_PIl 3.141592653589793238462643383279502884L

typedef double myf;
typedef int myi;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

myf fun(myf x) {
    return 4.0 / (1.0 + x * x);
}

myf rectangle_method(myf a, myf b, myi n) {
    myf deltax = (b - a) / n;
    myf res = 0.0;
    for (myi i = 0; i < n; i++)
        res += deltax * fun(a + deltax * 0.5 + i * deltax);
    return res;
}

void usage(char *argv0) {
    if (argv0)
        printf("Usage: mpirun -np <num_procs> %s <num_of_intervals>\n", argv0);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return -1;
    }

    myi nint = (myi)atoll(argv[1]);
    if (nint % size != 0) {
        if (rank == 0)
            printf("Number of intervals should be divisible by number of processes.\n");
        MPI_Finalize();
        return -1;
    }

    myi local_n = nint / size;
    myf a = 0.0, b = 1.0;
    myf local_a = a + rank * (b - a) / size;
    myf local_b = local_a + (b - a) / size;

    double start = MPI_Wtime();
    myf local_result = rectangle_method(local_a, local_b, local_n);
    myf global_result = 0.0;

    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Result: %16.14f\n", (double)global_result);
        printf("Error: %16.14f\n", (double)(global_result - M_PIl));
        printf("Elapsed time: %10.6f seconds\n", end - start);
    }

    MPI_Finalize();
    return 0;
}
