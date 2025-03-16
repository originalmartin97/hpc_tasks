#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>


#define MAXIMUM_ARRAY_SIZE 1000000


int main( int argc, char ** argv ) {
	int N = 0;
	int rank, size;

	if(argc == 2) {

		N = atoi(argv[1]);

		MPI_Init( &argc, &argv );
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		// Allocate memory for N sized array of doubles;
		double *array = (double *)malloc(N * sizeof(double));

		double sum1 = 0.0, sum2 = 0.0;

		// Randomize seed for each rank;
		srand( time(NULL) + rank );

		for(int i = 0; i < N; i++) {
			array[i] = (double)rand() / RAND_MAX * N;
			sum1 += array[i];
		}

		for(int i = N - 1; i > 0; i--){
			sum2 += array[i];
		}

		printf("Rank %d: sum of all elements (start-end): %f\n", rank, sum1);
		printf("Rank %d: sum of all elements (end-start): %f\n", rank, sum2);

		free(array);

		MPI_Finalize();

		return 0;
	} else {
		printf("The given arguments for the program were not handled!\n");
		printf("Please provide the syntax correctly.\n");
		printf("\t./<program executable> <P> <N>\n");
		printf("\t\t-P = number of ranks\n");
		printf("\t\t-N = number of size\n");

		return 1;
	}

	return 0;
}
