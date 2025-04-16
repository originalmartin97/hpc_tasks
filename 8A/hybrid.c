#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int main ( int argc, char *argv[] ){
        const long int N = 10000000000;
        const double h = 1.0/N;
        const double PI = 3.141592653589793238462643;
        double x, sum, pi, error, time, mypi; int i;

	int myrank, nproc;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        clock_t start_time, end_time;

        start_time = clock();

        sum = 0.0;

	#pragma omp parallel for shared(N, h, myrank, nproc), private(i,x), reduction(+:sum)
        for ( i = myrank; i <= N; i=i+nproc ){
                x = h * (double)i;
                sum += 4.0/( 1.0 + x * x );
        }

        mypi = h * sum;
	MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time = clock();

        time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	if ( myrank == 0 ){
		error = pi - PI;
        	error = error < 0 ? -error:error;
        	printf("pi = %18.16f +/- %18.16f\n", pi, error);
        	printf("time = %18.16f sec\n", time);
	}

	MPI_Finalize();
        return 0;
}
